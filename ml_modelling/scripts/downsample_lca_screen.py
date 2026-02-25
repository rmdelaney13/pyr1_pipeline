#!/usr/bin/env python3
"""Downsample LCA_screen negatives for more balanced training data.

Strategy:
  - Remove all 59R LCA_screen rows (493 rows, 0 binders)
  - Randomly sample ~150 from each of 59V, 59L, 59A in LCA_screen
  - Keep all other sources untouched
"""

import argparse
import re
import pandas as pd


def get_pos59_residue(signature: str) -> str:
    """Extract the residue at position 59 from a variant signature."""
    match = re.search(r'(?:^|;)59([A-Z])', str(signature))
    return match.group(1) if match else "wt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Input all_features.csv")
    parser.add_argument("--out", required=True, help="Output downsampled CSV")
    parser.add_argument("--sample-per-group", type=int, default=150,
                        help="Max rows to keep per pos59 group in LCA_screen (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n_orig = len(df)

    # Split LCA_screen from everything else
    is_lca_screen = df["label_source"] == "LCA_screen"
    df_other = df[~is_lca_screen].copy()
    df_lca = df[is_lca_screen].copy()

    # Annotate position 59
    df_lca["pos59"] = df_lca["variant_signature"].apply(get_pos59_residue)

    print(f"Original dataset: {n_orig} rows ({is_lca_screen.sum()} LCA_screen)")
    print(f"\nLCA_screen by pos59:")
    for res, grp in df_lca.groupby("pos59"):
        n_bind = (grp["label"] >= 0.5).sum()
        print(f"  59{res}: {len(grp)} rows, {n_bind} binders")

    # Remove 59R entirely
    df_lca_no_r = df_lca[df_lca["pos59"] != "R"].copy()
    n_removed_r = is_lca_screen.sum() - len(df_lca_no_r)
    print(f"\nRemoved 59R: {n_removed_r} rows")

    # Downsample remaining groups
    sampled_parts = []
    for res, grp in df_lca_no_r.groupby("pos59"):
        # Always keep binders
        binders = grp[grp["label"] >= 0.5]
        non_binders = grp[grp["label"] < 0.5]

        if len(non_binders) > args.sample_per_group:
            non_binders = non_binders.sample(n=args.sample_per_group,
                                              random_state=args.seed)
        sampled = pd.concat([binders, non_binders])
        print(f"  59{res}: {len(grp)} -> {len(sampled)} (kept {len(binders)} binders + {len(non_binders)} non-binders)")
        sampled_parts.append(sampled)

    df_lca_sampled = pd.concat(sampled_parts)
    df_lca_sampled = df_lca_sampled.drop(columns=["pos59"])

    # Recombine
    df_out = pd.concat([df_other, df_lca_sampled]).sort_index()

    # Summary
    print(f"\nFinal dataset: {len(df_out)} rows (was {n_orig})")
    print(f"  Removed: {n_orig - len(df_out)} LCA_screen rows")
    print(f"\nSource breakdown:")
    for src, grp in df_out.groupby("label_source"):
        n_bind = (grp["label"] >= 0.5).sum()
        print(f"  {src:25s}: {len(grp):5d} rows, {n_bind:4d} binders ({100*n_bind/len(grp):.1f}%)")

    total_bind = (df_out["label"] >= 0.5).sum()
    print(f"\n  Total: {len(df_out)} rows, {total_bind} binders ({100*total_bind/len(df_out):.1f}%)")

    df_out.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
