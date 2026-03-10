import argparse
import glob
import json
import os
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Two possible column schemas:
# Legacy (ligand-specific): dG_sep, buried_unsatisfied_polars, O1_polar_contact, O2_polar_contact, charge_satisfied
# General pipeline:         dG_sep, lig_unsat_polars, charge_satisfied, interface_unsats

REQUIRED_COLS_ALWAYS = ["filename", "dG_sep"]


def require_cols(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS_ALWAYS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")


def detect_schema(df: pd.DataFrame) -> str:
    """Detect whether we have legacy or general column schema."""
    if "buried_unsatisfied_polars" in df.columns:
        return "legacy"
    elif "lig_unsat_polars" in df.columns:
        return "general"
    else:
        raise ValueError(
            "Cannot detect schema: need either 'buried_unsatisfied_polars' or 'lig_unsat_polars' column"
        )


def yes_mask(series: pd.Series) -> pd.Series:
    """Matches 'yes', 'true', '1', '1.0' case-insensitive."""
    return series.astype(str).str.strip().str.lower().isin(["yes", "true", "1", "1.0"])


def pdb_name_from_csv_filename(csv_filename: str) -> str:
    base = os.path.basename(str(csv_filename)).strip()
    stem, _ext = os.path.splitext(base)
    # Remove common suffixes if present
    if stem.endswith("_relaxed_score"):
        stem = stem[:-len("_relaxed_score")]
    if stem.endswith("_score"):
        stem = stem[:-len("_score")]
    return f"{stem}.pdb"


def get_parent_dock(filename: str) -> str:
    """
    Extracts the parent dock name from the filename.
    For general pipeline: 'cluster_0001_a0000_rep_0_3.sc' -> 'cluster_0001_a0000_rep_0'
    For legacy: 'arrayXXX_passY_repackedZ_design_N.sc' -> prefix before '_design_'
    """
    name = str(filename)
    # General pipeline format: cluster_XXXX_aYYYY_rep_Z_designID.sc
    # The parent is everything up to the last _N (design number)
    if "_design_" in name:
        return name.split("_design_")[0]
    # General format: strip the last _N part (design index like _1, _2, _native)
    stem = os.path.splitext(name)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].isdigit() or parts[1] == "native"):
        return parts[0]
    return stem


def _parse_pdb_oxygens(pdb_path: str):
    """Extract ligand oxygen and water oxygen coordinates from a docked PDB."""
    PROTEIN_RES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
    WATER_RES = {"HOH", "TP3", "WAT", "TIP", "TP3W"}
    lig_oxygens = {}
    water_oxygens = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            if res_name in WATER_RES:
                if atom_name == "O":
                    water_oxygens.append(np.array([x, y, z]))
                continue
            if res_name not in PROTEIN_RES and atom_name in ("O1", "O2", "O3", "O4"):
                lig_oxygens[atom_name] = np.array([x, y, z])
    return lig_oxygens, water_oxygens


def classify_parent_docks(pdb_dir: str) -> Dict[str, str]:
    """Map each parent dock PDB to the oxygen class closest to water.

    Returns dict like {"cluster_0001_a0000_rep_0": "O3", ...}
    """
    mapping = {}
    for pdb_path in sorted(glob.glob(os.path.join(pdb_dir, "*.pdb"))):
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        lig_oxygens, water_oxygens = _parse_pdb_oxygens(pdb_path)
        if not lig_oxygens or not water_oxygens:
            continue
        distances = {}
        for name, lig_xyz in lig_oxygens.items():
            distances[name] = min(np.linalg.norm(lig_xyz - w) for w in water_oxygens)
        mapping[stem] = min(distances, key=distances.get)
    return mapping


def quota_select(candidates: pd.DataFrame, quotas: Dict[str, int],
                 max_per_parent: int) -> pd.DataFrame:
    """Select designs with per-oxygen-class quotas and surplus redistribution.

    candidates must have 'oxygen_class', 'parent_dock', and 'dG_sep' columns.
    """
    selected_parts = []
    total_surplus = 0

    # Track indices already selected for the redistribution pass
    selected_indices = set()

    for oxy_class in sorted(quotas.keys()):
        quota = quotas[oxy_class]
        class_df = candidates[candidates["oxygen_class"] == oxy_class].copy()
        if class_df.empty:
            total_surplus += quota
            print(f"  {oxy_class}:   0/{quota} (no candidates)")
            continue

        # Apply max_per_parent within this class
        class_df = class_df.sort_values("dG_sep", ascending=True)
        class_capped = class_df.groupby("parent_dock").head(max_per_parent)
        class_capped = class_capped.sort_values("dG_sep", ascending=True)

        taken = class_capped.head(quota)
        n_taken = len(taken)
        surplus = quota - n_taken
        total_surplus += surplus

        status = "quota filled" if surplus == 0 else f"{surplus} surplus to redistribute"
        print(f"  {oxy_class}: {n_taken:>4d}/{quota} ({status})")

        selected_parts.append(taken)
        selected_indices.update(taken.index)

    # Redistribution pass: fill surplus from best remaining candidates globally
    if total_surplus > 0 and len(selected_indices) < len(candidates):
        remaining = candidates[~candidates.index.isin(selected_indices)]
        # Apply max_per_parent across the bonus pool too
        remaining = remaining.sort_values("dG_sep", ascending=True)
        remaining_capped = remaining.groupby("parent_dock").head(max_per_parent)
        remaining_capped = remaining_capped.sort_values("dG_sep", ascending=True)
        bonus = remaining_capped.head(total_surplus)
        print(f"  Bonus (redistributed): {len(bonus)}")
        selected_parts.append(bonus)

    if not selected_parts:
        return pd.DataFrame(columns=candidates.columns)

    result = pd.concat(selected_parts, ignore_index=False)
    result = result.sort_values("dG_sep", ascending=True).reset_index(drop=True)
    print(f"  Total selected: {len(result)}")
    return result


def filter_designs(df: pd.DataFrame, target_n: int, max_unsat: int, max_per_parent: int,
                   check_o1: bool, check_o2: bool, check_charge: bool,
                   oxygen_quotas: Optional[Dict[str, int]] = None,
                   docked_pdb_dir: Optional[str] = None) -> pd.DataFrame:
    require_cols(df)
    schema = detect_schema(df)
    d = df.copy()

    # Determine the unsatisfied polars column name
    unsat_col = "buried_unsatisfied_polars" if schema == "legacy" else "lig_unsat_polars"

    # 1. Clean and Convert Data Types
    d["dG_sep"] = pd.to_numeric(d["dG_sep"], errors="coerce")
    d[unsat_col] = pd.to_numeric(d[unsat_col], errors="coerce")
    d = d.dropna(subset=["dG_sep", unsat_col]).copy()
    d = d.drop_duplicates(subset=["filename"], keep="first")

    # 2. Apply Dynamic Filters
    mask_combined = pd.Series([True] * len(d), index=d.index)

    if schema == "legacy":
        # Legacy schema has per-atom polar contact columns
        if check_o1 and "O1_polar_contact" in d.columns:
            mask_combined &= yes_mask(d["O1_polar_contact"])
        if check_o2 and "O2_polar_contact" in d.columns:
            mask_combined &= yes_mask(d["O2_polar_contact"])
    # General schema doesn't have O1/O2 columns - skip those checks

    if check_charge and "charge_satisfied" in d.columns:
        mask_combined &= yes_mask(d["charge_satisfied"])

    mask_unsats = d[unsat_col] <= max_unsat

    valid_candidates = d[mask_combined & mask_unsats].copy()

    # 3. Sort by fewest unsatisfied polars first, then best dG_sep
    #    This prioritizes designs with more ligand oxygens making H-bonds
    valid_candidates = valid_candidates.sort_values(
        [unsat_col, "dG_sep"], ascending=[True, True]
    )

    # 4. Identify Parent Dock
    valid_candidates["parent_dock"] = valid_candidates["filename"].apply(get_parent_dock)

    # 5. Oxygen-class quota selection OR global score ranking
    if oxygen_quotas and docked_pdb_dir:
        print(f"\nClassifying parent docks by oxygen class from: {docked_pdb_dir}")
        oxy_mapping = classify_parent_docks(docked_pdb_dir)
        print(f"  Classified {len(oxy_mapping)} parent docks")

        valid_candidates["oxygen_class"] = valid_candidates["parent_dock"].map(oxy_mapping)
        n_unmapped = valid_candidates["oxygen_class"].isna().sum()
        if n_unmapped > 0:
            print(f"  Warning: {n_unmapped} designs have unmapped parent docks (excluded)")
            valid_candidates = valid_candidates.dropna(subset=["oxygen_class"]).copy()

        print("\nOxygen-class quota selection:")
        final_set = quota_select(valid_candidates, oxygen_quotas, max_per_parent)
    else:
        # Original behavior: global score ranking with max_per_parent cap
        balanced_candidates = valid_candidates.groupby("parent_dock").head(max_per_parent)
        final_set = balanced_candidates.sort_values("dG_sep", ascending=True).head(target_n)

    return final_set.reset_index(drop=True)


def copy_pdbs(df: pd.DataFrame, src_dir: str, out_dir: str) -> Tuple[int, int]:
    copied, missing = 0, 0
    for _, row in df.iterrows():
        pdb_name = pdb_name_from_csv_filename(row["filename"])
        src = os.path.join(src_dir, pdb_name)
        dst = os.path.join(out_dir, pdb_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    return copied, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter designs with configurable polar contact requirements.")
    parser.add_argument("input_csv", help="Input relax CSV")
    parser.add_argument("relax_dir", help="Directory containing relaxed PDBs")
    parser.add_argument("output_dir", help="Output directory for filtered results")
    parser.add_argument("--target_n", type=int, default=500, help="Final number of designs to output")
    parser.add_argument("--max_unsat", type=int, default=5, help="Maximum allowed unsatisfied ligand polars")
    parser.add_argument("--max_per_parent", type=int, default=5, help="Maximum designs allowed per parent dock structure")
    parser.add_argument("--output_csv_name", default="filtered.csv")
    parser.add_argument("--no_copy_pdbs", action="store_true", help="Do not copy PDBs")

    # Filter Flags (only apply to legacy schema with O1/O2 columns)
    parser.add_argument("--ignore_o1", action="store_true", help="If set, O1 polar contact is NOT required")
    parser.add_argument("--ignore_o2", action="store_true", help="If set, O2 polar contact is NOT required")
    parser.add_argument("--ignore_charge", action="store_true", help="If set, charge satisfaction is NOT required")

    # Oxygen-class quota selection (optional)
    parser.add_argument("--oxygen_quotas", type=str, default=None,
                        help='JSON dict of oxygen class quotas, e.g. \'{"O3":500,"O2":250,"O1":250,"O4":100}\'')
    parser.add_argument("--docked_pdb_dir", type=str, default=None,
                        help="Path to clustered docked PDB directory (required with --oxygen_quotas)")

    args = parser.parse_args()

    if args.target_n <= 0:
        raise ValueError("--target_n must be > 0")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    schema = detect_schema(df)

    # Determine requirements based on flags
    req_o1 = not args.ignore_o1
    req_o2 = not args.ignore_o2
    req_charge = not args.ignore_charge

    print(f"Detected schema: {schema}")
    print(f"Filtering: max {args.max_per_parent} per parent, max {args.max_unsat} unsats...")
    print(f"Criteria Active:")
    if schema == "legacy":
        print(f"  - O1 Required: {req_o1}")
        print(f"  - O2 Required: {req_o2}")
    else:
        print(f"  - O1/O2 checks: N/A (general schema)")
    print(f"  - Charge Required: {req_charge}")

    # Parse oxygen quotas if provided
    oxy_quotas = None
    if args.oxygen_quotas:
        oxy_quotas = json.loads(args.oxygen_quotas)
        if not args.docked_pdb_dir:
            raise ValueError("--docked_pdb_dir is required when --oxygen_quotas is set")
        print(f"  - Oxygen quotas: {oxy_quotas}")
        print(f"  - Docked PDB dir: {args.docked_pdb_dir}")

    selected = filter_designs(df,
                              target_n=args.target_n,
                              max_unsat=args.max_unsat,
                              max_per_parent=args.max_per_parent,
                              check_o1=req_o1,
                              check_o2=req_o2,
                              check_charge=req_charge,
                              oxygen_quotas=oxy_quotas,
                              docked_pdb_dir=args.docked_pdb_dir)

    out_csv = os.path.join(args.output_dir, args.output_csv_name)
    selected.to_csv(out_csv, index=False)

    # --- Stats ---
    num_designs = len(selected)
    unique_docks = selected["parent_dock"].nunique() if "parent_dock" in selected else 0

    print(f"\nWrote filtered CSV to: {out_csv}")
    print(f"Total designs selected: {num_designs}")
    print(f"Unique parent docks represented: {unique_docks}")

    if args.no_copy_pdbs:
        print("Skipping PDB copying (--no_copy_pdbs set)")
        return

    copied, missing = copy_pdbs(selected, args.relax_dir, args.output_dir)
    print(f"PDBs copied: {copied}")
    if missing:
        print(f"PDBs missing (not found in relax_dir): {missing}")


if __name__ == "__main__":
    main()
