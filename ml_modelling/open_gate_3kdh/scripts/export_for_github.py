#!/usr/bin/env python3
"""Export open-gate structures and guide CSV for GitHub sharing.

Copies the best threaded_relaxed PDBs into a clean directory and generates
a guide CSV matching the format used for the Boltz closed-gate structures.

Usage:
    python export_for_github.py \
        --relaxed-dir outputs/threaded_relaxed/ \
        --csv ../analysis/boltz_LCA/md_candidates_lca_top100.csv \
        --qc-csv outputs/qc_report.csv \
        --ligand-rmsd-csv outputs/ligand_rmsd.csv \
        --output-dir outputs/open_gate_pdbs
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


GROUP_DESCRIPTIONS = {
    "binder": "Experimentally validated binder (Y2H/FACS). Open-gate (3KDH backbone) structure with ligand placed before relax.",
    "non_binder": "Did not show binding experimentally, but predicted as high-quality complex by Boltz2. Open-gate structure for MD comparison.",
    "negative_low_pocket": "Predicted with correct ligand placement but low pocket confidence. Open-gate structure for MD negative control.",
    "negative_fail_gate": "Predicted with low-confidence ligand placement or incorrect H-bonding. Open-gate structure for MD negative control.",
}


def main():
    parser = argparse.ArgumentParser(
        description="Export open-gate structures for GitHub"
    )
    parser.add_argument("--relaxed-dir", required=True)
    parser.add_argument("--csv", required=True,
                        help="MD candidates CSV (with pair_id, variant_signature, label, etc.)")
    parser.add_argument("--qc-csv", help="QC report CSV")
    parser.add_argument("--ligand-rmsd-csv", help="Ligand RMSD CSV")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    relaxed_dir = Path(args.relaxed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Load QC if available
    qc_df = None
    if args.qc_csv and Path(args.qc_csv).exists():
        qc_df = pd.read_csv(args.qc_csv)

    # Load ligand RMSD if available
    rmsd_df = None
    if args.ligand_rmsd_csv and Path(args.ligand_rmsd_csv).exists():
        rmsd_df = pd.read_csv(args.ligand_rmsd_csv)

    rows = []
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        pair_id = row["pair_id"]
        src = relaxed_dir / f"{pair_id}_threaded_relaxed.pdb"

        if not src.exists():
            print(f"  MISSING: {pair_id}")
            missing += 1
            continue

        # Copy with clean name
        dst = output_dir / f"{pair_id}_open_gate.pdb"
        shutil.copy2(src, dst)
        copied += 1

        # Determine category
        label = row.get("label", "")
        category = row.get("category", "")
        if not category:
            if label == 1 or label == "1":
                category = "binder"
            else:
                label_tier = str(row.get("label_tier", ""))
                if "low_pocket" in label_tier:
                    category = "negative_low_pocket"
                elif "fail_gate" in label_tier:
                    category = "negative_fail_gate"
                else:
                    category = "non_binder"

        guide_row = {
            "pair_id": pair_id,
            "md_group": category,
            "label": row.get("label", ""),
            "variant_name": row.get("variant_name", ""),
            "variant_signature": row.get("variant_signature", ""),
            "label_tier": row.get("label_tier", ""),
            "pdb_file": f"{pair_id}_open_gate.pdb",
            "structure_type": "open_gate_3kdh",
            "group_description": GROUP_DESCRIPTIONS.get(category, ""),
        }

        # Add QC metrics if available
        if qc_df is not None:
            qc_row = qc_df[qc_df["pair_id"] == pair_id]
            if not qc_row.empty:
                qc_row = qc_row.iloc[0]
                guide_row["qc_status"] = qc_row.get("status", "")
                guide_row["backbone_rmsd"] = qc_row.get("backbone_rmsd_to_3kdh", "")
                guide_row["gate_rmsd"] = qc_row.get("gate_rmsd_to_3kdh", "")
                guide_row["n_severe_clashes"] = qc_row.get("n_severe_clashes", "")
                guide_row["min_lig_prot_distance"] = qc_row.get("min_lig_prot_distance", "")

        # Add ligand RMSD if available
        if rmsd_df is not None:
            rmsd_row = rmsd_df[rmsd_df["pair_id"] == pair_id]
            if not rmsd_row.empty:
                guide_row["ligand_rmsd_to_boltz"] = rmsd_row.iloc[0].get("ligand_rmsd", "")

        rows.append(guide_row)

    guide_df = pd.DataFrame(rows)
    guide_path = output_dir / "open_gate_guide.csv"
    guide_df.to_csv(guide_path, index=False)

    print(f"\nExported {copied} structures to {output_dir}")
    if missing:
        print(f"  {missing} structures missing")
    print(f"Guide CSV: {guide_path}")
    print(f"\nCategory breakdown:")
    print(guide_df["md_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
