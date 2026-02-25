#!/usr/bin/env python3
"""
Backfill affinity_uM into master_pairs.csv from source data files.

Sources:
  - WIN SSM:  affinity_data_win.csv  (Kd in nM) — already done during pair generation
  - PNAS:     pnas_data.csv          (min_conc: 1, 10, 100, or 'nd')
  - PNAS v2:  pnas_data_2.csv        (finer: 0.025–100 uM, preferred when available)

The PNAS concentration data represents the minimum concentration (in uM) at which
binding was detected.  For pnas_data_2.csv, we compute min_conc from the individual
concentration columns (lowest concentration with '+' response).

Usage:
    python backfill_affinity.py                      # dry-run
    python backfill_affinity.py --write               # write updated master_pairs.csv
    python backfill_affinity.py --write --prefer-v2   # use pnas_data_2.csv where available
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# WT residues at each PYR1 pocket position (used to build variant signatures)
# Format: {position_number: WT_amino_acid_1letter}
WT_RESIDUES = {
    59: "K", 65: "C", 71: "F", 74: "R", 81: "V", 83: "V",
    87: "L", 89: "A", 92: "S", 94: "E", 108: "F", 109: "S",
    110: "I", 117: "L", 120: "Y", 122: "S", 124: "T", 134: "R",
    141: "E", 159: "F", 160: "A", 163: "V", 164: "V", 167: "N",
    178: "M", 184: "D",
}

# Map column header -> position number
# pnas_data.csv: "K59" -> 59, "V81" -> 81, etc.
# pnas_data_2.csv: same format but more positions


def _parse_mutation_columns(row, mutation_cols):
    """
    Build a variant_signature from PNAS mutation columns.

    Each column header is like 'K59', 'V81', etc. (WT_AA + position).
    The value is the mutant AA (single letter), or empty/NaN if WT.

    Returns signature like '89G;122M;160L' (sorted by position).
    """
    parts = []
    for col in mutation_cols:
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            continue
        # Extract position from column name (e.g., 'K59' -> 59)
        pos_str = "".join(c for c in col if c.isdigit())
        if not pos_str:
            continue
        pos = int(pos_str)
        mutant_aa = str(val).strip().upper()
        if len(mutant_aa) == 1 and mutant_aa.isalpha():
            parts.append((pos, f"{pos}{mutant_aa}"))

    parts.sort(key=lambda x: x[0])
    return ";".join(p[1] for p in parts)


def _compute_min_conc_v2(row, conc_cols):
    """
    Compute minimum active concentration from pnas_data_2.csv.

    conc_cols: list of (column_name, concentration_value) sorted by concentration.
    Returns the lowest concentration with '+' response, or 'nd' if none.
    """
    for col_name, conc_val in conc_cols:
        val = row.get(col_name)
        if isinstance(val, str) and val.strip() == "+":
            return conc_val
    return "nd"


def load_pnas_v1(path):
    """Load pnas_data.csv and build (ligand_name_lower, signature) -> min_conc mapping."""
    df = pd.read_csv(path)

    # Identify mutation columns (format: single_letter + digits, like K59, V81)
    mutation_cols = [c for c in df.columns
                     if len(c) >= 2 and c[0].isalpha() and c[1:].isdigit()]

    records = {}
    for _, row in df.iterrows():
        ligand = str(row["library_name"]).strip().lower()
        sig = _parse_mutation_columns(row, mutation_cols)
        min_conc = row.get("min_conc")
        if pd.isna(min_conc):
            continue
        min_conc = str(min_conc).strip()
        records[(ligand, sig)] = min_conc

    return records


def load_pnas_v2(path):
    """Load pnas_data_2.csv and build (ligand_name_lower, signature) -> min_conc mapping."""
    df = pd.read_csv(path)

    # Identify mutation columns
    mutation_cols = [c for c in df.columns
                     if len(c) >= 2 and c[0].isalpha() and c[1:].isdigit()]

    # Identify concentration columns (numeric headers)
    conc_cols = []
    for c in df.columns:
        try:
            val = float(c)
            conc_cols.append((c, val))
        except ValueError:
            continue
    conc_cols.sort(key=lambda x: x[1])  # sort by concentration ascending

    records = {}
    for _, row in df.iterrows():
        ligand = str(row["compound"]).strip().lower()
        sig = _parse_mutation_columns(row, mutation_cols)
        min_conc = _compute_min_conc_v2(row, conc_cols)
        records[(ligand, sig)] = min_conc

    return records


def main():
    parser = argparse.ArgumentParser(description="Backfill affinity_uM in master_pairs.csv")
    parser.add_argument("--master-pairs", type=Path,
                        default=DATA_DIR / "master_pairs.csv",
                        help="Path to master_pairs.csv")
    parser.add_argument("--write", action="store_true",
                        help="Write updated master_pairs.csv (default: dry-run)")
    parser.add_argument("--prefer-v2", action="store_true",
                        help="Prefer pnas_data_2.csv finer concentrations over pnas_data.csv")
    args = parser.parse_args()

    # Load master_pairs
    mp = pd.read_csv(args.master_pairs)
    print(f"Loaded {len(mp)} rows from {args.master_pairs.name}")

    # Count existing affinity data
    has_affinity = mp["affinity_uM"].notna() & (mp["affinity_uM"] != "")
    print(f"  Already have affinity_uM: {has_affinity.sum()} rows")

    # Load PNAS source data
    pnas_v1_path = DATA_DIR / "pnas_data.csv"
    pnas_v2_path = DATA_DIR / "pnas_data_2.csv"

    pnas_v1 = load_pnas_v1(pnas_v1_path) if pnas_v1_path.exists() else {}
    pnas_v2 = load_pnas_v2(pnas_v2_path) if pnas_v2_path.exists() else {}
    print(f"  PNAS v1 records: {len(pnas_v1)}")
    print(f"  PNAS v2 records: {len(pnas_v2)}")

    # Build combined lookup: prefer v2 if --prefer-v2
    if args.prefer_v2:
        pnas_lookup = {**pnas_v1, **pnas_v2}  # v2 overwrites v1
        print("  Using pnas_data_2.csv where available (--prefer-v2)")
    else:
        pnas_lookup = {**pnas_v2, **pnas_v1}  # v1 overwrites v2
        print("  Using pnas_data.csv as primary (use --prefer-v2 for finer data)")

    # Match PNAS rows
    pnas_mask = mp["label_source"] == "pnas_cutler"
    pnas_rows = mp[pnas_mask]
    print(f"\n  PNAS rows in master_pairs: {len(pnas_rows)}")

    matched = 0
    unmatched = 0
    already_filled = 0
    nd_count = 0
    conc_distribution = {}

    for idx in pnas_rows.index:
        ligand = str(mp.at[idx, "ligand_name"]).strip()
        sig = str(mp.at[idx, "variant_signature"]).strip()

        # Check if already has affinity
        current = mp.at[idx, "affinity_uM"]
        if pd.notna(current) and str(current).strip() != "":
            already_filled += 1
            continue

        # Look up in PNAS data (case-insensitive ligand name)
        key = (ligand.lower(), sig)
        if key in pnas_lookup:
            min_conc = pnas_lookup[key]
            if min_conc == "nd" or min_conc == "ND":
                nd_count += 1
                # nd = not detected at any concentration -> non-binder
                # Leave affinity_uM empty (or could set to a high value)
                continue
            try:
                conc_val = float(min_conc)
                mp.at[idx, "affinity_uM"] = conc_val
                matched += 1
                conc_distribution[conc_val] = conc_distribution.get(conc_val, 0) + 1
            except ValueError:
                unmatched += 1
        else:
            unmatched += 1

    print(f"\n  Results:")
    print(f"    Already filled: {already_filled}")
    print(f"    Matched & filled: {matched}")
    print(f"    Not detected (nd): {nd_count}")
    print(f"    Unmatched: {unmatched}")

    if conc_distribution:
        print(f"\n  Concentration distribution of backfilled values:")
        for conc in sorted(conc_distribution.keys()):
            print(f"    {conc:>8.3f} uM: {conc_distribution[conc]} pairs")

    # Show some unmatched examples for debugging
    if unmatched > 0:
        print(f"\n  First 5 unmatched PNAS pairs:")
        count = 0
        for idx in pnas_rows.index:
            if count >= 5:
                break
            ligand = str(mp.at[idx, "ligand_name"]).strip()
            sig = str(mp.at[idx, "variant_signature"]).strip()
            current = mp.at[idx, "affinity_uM"]
            if pd.notna(current) and str(current).strip() != "":
                continue
            key = (ligand.lower(), sig)
            if key not in pnas_lookup:
                print(f"    ligand='{ligand}', sig='{sig}'")
                # Show closest matches
                close = [k for k in pnas_lookup if k[1] == sig]
                if close:
                    print(f"      -> same sig, different ligand: {[k[0] for k in close[:3]]}")
                close = [k for k in pnas_lookup if k[0] == ligand]
                if close:
                    print(f"      -> same ligand, different sig: {[k[1] for k in close[:3]]}")
                count += 1

    # Summary of all affinity data
    total_with_affinity = mp["affinity_uM"].notna().sum()
    # Count non-empty strings too
    total_with_affinity = sum(
        1 for v in mp["affinity_uM"]
        if pd.notna(v) and str(v).strip() != ""
    )
    print(f"\n  Total rows with affinity_uM after backfill: {total_with_affinity}")
    print(f"    WIN SSM: {sum(1 for i in mp[mp['label_source'] == 'win_ssm'].index if pd.notna(mp.at[i, 'affinity_uM']) and str(mp.at[i, 'affinity_uM']).strip() != '')}")
    print(f"    PNAS:    {sum(1 for i in mp[mp['label_source'] == 'pnas_cutler'].index if pd.notna(mp.at[i, 'affinity_uM']) and str(mp.at[i, 'affinity_uM']).strip() != '')}")

    if args.write:
        mp.to_csv(args.master_pairs, index=False)
        print(f"\n  WRITTEN: {args.master_pairs}")
    else:
        print(f"\n  DRY RUN — use --write to save changes")


if __name__ == "__main__":
    main()
