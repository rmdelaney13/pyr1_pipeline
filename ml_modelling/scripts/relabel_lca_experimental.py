#!/usr/bin/env python3
"""Move unconjugated Lithocholic Acid binders from experimental (Tier 1)
to LCA_screen (Tier 4) so per-source plots show balanced binder/non-binder
distributions for LCA_screen.

Only moves "Lithocholic Acid" — keeps GlycoLithocholic Acid and
Lithocholic Acid 3 -S in experimental.

Usage:
    python relabel_lca_experimental.py \
        --features ml_modelling/data/all_features.csv \
        [--balanced ml_modelling/data/all_features_balanced.csv] \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="Path to all_features.csv")
    parser.add_argument("--balanced", default=None,
                        help="Optional path to all_features_balanced.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without modifying files")
    args = parser.parse_args()

    feature_files = [args.features]
    if args.balanced:
        feature_files.append(args.balanced)

    for fpath in feature_files:
        print(f"\n{'='*60}")
        print(f"Processing: {fpath}")
        print(f"{'='*60}")

        df = pd.read_csv(fpath)

        # Find unconjugated LCA experimental binders
        # Exact match on "Lithocholic Acid" — excludes "Lithocholic Acid 3 -S"
        # and "GlycoLithocholic Acid"
        mask = ((df["ligand_name"] == "Lithocholic Acid") &
                (df["label_source"] == "experimental"))

        n_matched = mask.sum()
        n_binders = int(df.loc[mask, "label"].ge(0.5).sum()) if n_matched > 0 else 0
        print(f"  Found {n_matched} unconjugated LCA experimental rows "
              f"({n_binders} binders)")

        if n_matched == 0:
            print("  Nothing to relabel.")
            continue

        # Show before state
        print(f"\n  Before relabeling:")
        for src in ["experimental", "LCA_screen"]:
            sub = df[df["label_source"] == src]
            lca_sub = sub[sub["ligand_name"].str.match(r"^Lithocholic", case=False, na=False)]
            n_b = int(lca_sub["label"].ge(0.5).sum())
            print(f"    {src:20s}: {len(sub):4d} total, {len(lca_sub):4d} LCA-family, "
                  f"{n_b:3d} binders")

        # Relabel
        df.loc[mask, "label_source"] = "LCA_screen"
        if "label_tier" in df.columns:
            df.loc[mask, "label_tier"] = "Tier 4"
        if "label_confidence" in df.columns:
            df.loc[mask, "label_confidence"] = 0.9

        # Show after state
        print(f"\n  After relabeling:")
        for src in ["experimental", "LCA_screen"]:
            sub = df[df["label_source"] == src]
            lca_sub = sub[sub["ligand_name"].str.match(r"^Lithocholic", case=False, na=False)]
            n_b = int(lca_sub["label"].ge(0.5).sum())
            print(f"    {src:20s}: {len(sub):4d} total, {len(lca_sub):4d} LCA-family, "
                  f"{n_b:3d} binders")

        # LCA_screen detail
        lca_screen = df[df["label_source"] == "LCA_screen"]
        n_b = int(lca_screen["label"].ge(0.5).sum())
        print(f"\n  LCA_screen now: {len(lca_screen)} pairs, {n_b} binders "
              f"({100*n_b/len(lca_screen):.1f}%)")

        if args.dry_run:
            print(f"\n  DRY RUN: no files modified")
        else:
            df.to_csv(fpath, index=False)
            print(f"\n  Saved: {fpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
