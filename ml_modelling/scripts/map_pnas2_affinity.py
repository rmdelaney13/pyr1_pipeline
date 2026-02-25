#!/usr/bin/env python3
"""Map affinity data from pnas_data_2.csv to all_features.csv.

Reads concentration-response data from pnas_data_2.csv, computes the minimum
concentration (uM) at which "+" appears for each variant-compound pair, then
maps that value into the affinity_EC50_uM column of all_features.csv (and
optionally all_features_balanced.csv).

Matching logic:
  - pnas_data_2 "name" column  ->  all_features "variant_name" column
  - pnas_data_2 "compound"     ->  all_features "ligand_name"
  - Only rows with label_source == "experimental" are candidates

Usage:
    python map_pnas2_affinity.py \
        --pnas2 ml_modelling/data/pnas_data_2.csv \
        --features ml_modelling/data/all_features.csv \
        [--balanced ml_modelling/data/all_features_balanced.csv] \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import pandas as pd
import sys


# Concentration columns in pnas_data_2.csv (in uM)
CONC_COLUMNS = ["0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "25", "50", "100"]
CONC_VALUES = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]


def compute_min_conc(row: pd.Series) -> float | None:
    """Return the lowest concentration (uM) where '+' was observed, or None."""
    for col, val in zip(CONC_COLUMNS, CONC_VALUES):
        if col in row.index and str(row[col]).strip() == "+":
            return val
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pnas2", required=True, help="Path to pnas_data_2.csv")
    parser.add_argument("--features", required=True, help="Path to all_features.csv")
    parser.add_argument("--balanced", default=None,
                        help="Optional path to all_features_balanced.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print mapping without modifying files")
    args = parser.parse_args()

    # --- Read pnas_data_2 ---
    pnas2 = pd.read_csv(args.pnas2)
    # Drop empty rows (trailing blank rows from Excel export)
    pnas2 = pnas2.dropna(subset=["name", "compound"], how="any")
    pnas2 = pnas2[pnas2["name"].str.strip() != ""]
    print(f"Read {len(pnas2)} rows from pnas_data_2.csv")

    # Compute min_conc for each row
    pnas2["min_conc_uM"] = pnas2.apply(compute_min_conc, axis=1)

    # Build lookup: (variant_name, compound) -> min_conc_uM
    lookup = {}
    for _, row in pnas2.iterrows():
        name = str(row["name"]).strip()
        compound = str(row["compound"]).strip()
        min_conc = row["min_conc_uM"]
        if min_conc is not None:
            lookup[(name, compound)] = min_conc

    print(f"\nBuilt lookup with {len(lookup)} entries with measurable affinity")
    print(f"  (out of {len(pnas2)} total rows; {len(pnas2) - len(lookup)} had no '+' at any concentration)\n")

    # Show summary by compound
    print("  Compound                       Variants  Min conc range (uM)")
    print("  " + "-" * 65)
    for compound in pnas2["compound"].unique():
        sub = pnas2[pnas2["compound"] == compound]
        concs = sub["min_conc_uM"].dropna()
        if len(concs) > 0:
            print(f"  {compound:30s}  {len(sub):3d}      {concs.min():.3f} - {concs.max():.3f}")
        else:
            print(f"  {compound:30s}  {len(sub):3d}      (no + observed)")

    # --- Process each features file ---
    feature_files = [args.features]
    if args.balanced:
        feature_files.append(args.balanced)

    for fpath in feature_files:
        print(f"\n{'='*60}")
        print(f"Processing: {fpath}")
        print(f"{'='*60}")

        df = pd.read_csv(fpath)
        n_total = len(df)

        # Only consider experimental rows as candidates
        is_experimental = df["label_source"] == "experimental"
        print(f"  Total rows: {n_total}, experimental: {is_experimental.sum()}")

        matched = 0
        already_had = 0
        not_found = []

        for idx in df[is_experimental].index:
            vname = str(df.at[idx, "variant_name"]).strip()
            lname = str(df.at[idx, "ligand_name"]).strip()
            key = (vname, lname)

            if key in lookup:
                old_val = df.at[idx, "affinity_EC50_uM"]
                df.at[idx, "affinity_EC50_uM"] = lookup[key]
                if pd.notna(old_val) and old_val != "" and float(old_val) > 0:
                    already_had += 1
                matched += 1

        # Check which pnas2 entries did NOT match any row
        for key in lookup:
            vname, compound = key
            mask = (df["variant_name"] == vname) & (df["ligand_name"] == compound)
            if mask.sum() == 0:
                not_found.append(key)

        print(f"  Matched: {matched} rows updated with affinity_EC50_uM")
        if already_had:
            print(f"    ({already_had} already had a value, now overwritten)")
        if not_found:
            print(f"  WARNING: {len(not_found)} pnas_data_2 entries not found in features:")
            for vn, cmp in not_found[:15]:
                print(f"    {vn} / {cmp}")
            if len(not_found) > 15:
                print(f"    ... and {len(not_found) - 15} more")

        # Show what was mapped
        mapped_rows = df[(is_experimental) & (df["affinity_EC50_uM"].notna()) &
                         (df["affinity_EC50_uM"] != "")]
        if len(mapped_rows) > 0:
            print(f"\n  Sample of mapped rows:")
            print(f"  {'pair_id':12s} {'ligand':25s} {'variant':12s} {'affinity_uM':>12s}")
            for _, r in mapped_rows.head(15).iterrows():
                print(f"  {r['pair_id']:12s} {r['ligand_name']:25s} "
                      f"{r['variant_name']:12s} {r['affinity_EC50_uM']:>12}")

        if args.dry_run:
            print(f"\n  DRY RUN: no files modified")
        else:
            df.to_csv(fpath, index=False)
            print(f"\n  Saved updated file: {fpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
