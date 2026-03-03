#!/usr/bin/env python3
"""
Analyze binary + ternary Boltz2 metrics for LCA training data.
Compute ROC-AUC for individual metrics and composite scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent
LABELS_DIR = RESULTS_DIR.parents[1] / "data" / "boltz_lca_conjugates"

LIGANDS = {
    "lca": "Lithocholic Acid",
    "glca": "GlycoLithocholic Acid",
    "lca3s": "Lithocholic Acid 3-S",
}


def load_merged_with_labels(ligand_key: str) -> pd.DataFrame:
    """Load merged results and join with labels."""
    merged = pd.read_csv(RESULTS_DIR / f"boltz_{ligand_key}_merged_results.csv")
    labels = pd.read_csv(LABELS_DIR / f"boltz_{ligand_key}_binary.csv")

    # Labels use pair_id, merged uses name
    label_map = labels.set_index("pair_id")["label"].to_dict()
    merged["label"] = merged["name"].map(label_map)

    # Drop rows without labels
    merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)

    # Derived metric: ternary - binary H-bond distance.
    # Positive = binder. In binary, binders have a ligand acceptor (OH or COO-)
    # correctly positioned near the conserved water. When HAB1 is added in ternary,
    # it compresses the pocket (Trp385 lock insertion), shifting the ligand away from
    # the 3QN1 reference water position → longer ternary distance.
    # Non-binders have the COO-to-R116 artifact in binary (carboxylate near R116,
    # which is adjacent to the water). HAB1 blocks R116 in ternary, so the non-binder
    # ligand drifts without a strong anchor → distance stays near zero or turns negative.
    merged["hbond_distance_delta"] = (
        pd.to_numeric(merged.get("ternary_hbond_distance"), errors="coerce")
        - pd.to_numeric(merged.get("binary_hbond_distance"), errors="coerce")
    )

    return merged


def compute_aucs(df: pd.DataFrame, metrics: list, higher_is_better: dict) -> dict:
    """Compute AUC for each metric."""
    results = {}
    for m in metrics:
        if m not in df.columns:
            continue
        vals = pd.to_numeric(df[m], errors="coerce")
        mask = vals.notna()
        if mask.sum() < 10:
            continue
        y = df.loc[mask, "label"].values
        x = vals[mask].values
        if len(np.unique(y)) < 2:
            continue
        try:
            auc = roc_auc_score(y, x)
            if not higher_is_better.get(m, True):
                auc = 1 - auc
            results[m] = auc
        except Exception:
            continue
    return results


def compute_composite_aucs(df: pd.DataFrame, metric_pairs: list, higher_is_better: dict) -> dict:
    """Compute AUC for Z-score composites of metric pairs."""
    results = {}
    for m1, m2 in metric_pairs:
        if m1 not in df.columns or m2 not in df.columns:
            continue
        v1 = pd.to_numeric(df[m1], errors="coerce")
        v2 = pd.to_numeric(df[m2], errors="coerce")
        mask = v1.notna() & v2.notna()
        if mask.sum() < 10:
            continue

        z1 = zscore(v1[mask])
        z2 = zscore(v2[mask])

        # Flip sign for lower-is-better metrics
        if not higher_is_better.get(m1, True):
            z1 = -z1
        if not higher_is_better.get(m2, True):
            z2 = -z2

        composite = z1 + z2
        y = df.loc[mask, "label"].values
        if len(np.unique(y)) < 2:
            continue
        try:
            auc = roc_auc_score(y, composite)
            results[f"Z({m1})+Z({m2})"] = auc
        except Exception:
            continue
    return results


# Metrics to evaluate
BINARY_METRICS = [
    "binary_iptm", "binary_ligand_iptm", "binary_confidence_score",
    "binary_complex_plddt", "binary_complex_iplddt", "binary_plddt_protein",
    "binary_plddt_ligand", "binary_plddt_pocket",
    "binary_affinity_probability_binary",
    "binary_hbond_distance", "binary_hbond_angle",
    "binary_boltz_score", "binary_total_score",
]

TERNARY_METRICS = [
    "ternary_iptm", "ternary_ligand_iptm", "ternary_protein_iptm",
    "ternary_confidence_score",
    "ternary_complex_plddt", "ternary_complex_iplddt",
    "ternary_plddt_protein", "ternary_plddt_ligand", "ternary_plddt_hab1",
    "ternary_plddt_pocket",
    "ternary_affinity_probability_binary",
    "ternary_hbond_distance", "ternary_hbond_angle",
    "ternary_boltz_score", "ternary_total_score",
    "ternary_trp211_ligand_distance",
]

# Pose consistency metrics (binary vs ternary comparison)
CONSISTENCY_METRICS = [
    "hbond_distance_delta",            # ternary - binary hbond dist (positive = binder)
    "ligand_rmsd_binary_vs_ternary",   # ligand flip between binary/ternary poses
    "pocket_rmsd_binary_vs_ternary",   # pocket deformation when HAB1 docks
    # COO-to-R116 artifact metrics (from compute_ligand_flip_metrics in analyze_boltz_output.py)
    # These specifically detect the binary-only COO-to-R116 salt bridge artifact.
    # NOTE: binary_flip_score and binary_oh_to_water_dist are intentionally EXCLUDED —
    # they assume OH-near-water is the "correct" orientation, which is not always true
    # (COO-near-water is also a valid binding mode). Only coo_to_r116_dist is artifact-specific.
    "binary_coo_to_r116_dist",         # carboxylate-to-R116 dist (higher = no R116 artifact)
    "binary_coo_to_water_dist",        # carboxylate-to-water dist (informational only)
    "ternary_coo_to_r116_dist",        # same for ternary (expect higher than binary on avg)
]

ALL_METRICS = BINARY_METRICS + TERNARY_METRICS + CONSISTENCY_METRICS

# Higher is better for most, except distance/angle (want close to ideal)
HIGHER_IS_BETTER = {m: True for m in ALL_METRICS}
# H-bond geometry: shorter binary distance = ligand acceptor (OH or COO-) near the
# conserved water = correct pocket interaction. Both orientations are valid;
# the COO-to-R116 artifact in binary gives longer distances because R116 is
# adjacent to (not at) the reference water position.
HIGHER_IS_BETTER["binary_hbond_distance"] = False
# BUG FIX: ternary hbond distance is HIGHER for binders.
# HAB1 compresses the pocket (Trp385 insertion) and shifts the ligand away from the
# 3QN1 reference water position → binders show larger ternary distances.
# Non-binders lose their COO-to-R116 anchor (blocked by HAB1) and drift → no consistent shift.
HIGHER_IS_BETTER["ternary_hbond_distance"] = True
HIGHER_IS_BETTER["hbond_distance_delta"] = True     # positive = binder (delta = ternary - binary)
HIGHER_IS_BETTER["binary_complex_pde"] = False
HIGHER_IS_BETTER["ternary_complex_pde"] = False
HIGHER_IS_BETTER["binary_complex_ipde"] = False
HIGHER_IS_BETTER["ternary_complex_ipde"] = False
HIGHER_IS_BETTER["ternary_trp211_ligand_distance"] = False
# Lower RMSD = more consistent pose = binder (lower is better)
HIGHER_IS_BETTER["ligand_rmsd_binary_vs_ternary"] = False
HIGHER_IS_BETTER["pocket_rmsd_binary_vs_ternary"] = False
# COO-to-R116 artifact metrics (orientation-agnostic: only detects the R116 salt bridge artifact)
HIGHER_IS_BETTER["binary_coo_to_r116_dist"] = True  # higher = COO far from R116 = no artifact
HIGHER_IS_BETTER["binary_coo_to_water_dist"] = True # informational; higher = COO far from water
HIGHER_IS_BETTER["ternary_coo_to_r116_dist"] = True

# Composite pairs to test
COMPOSITE_PAIRS = [
    # Binary-only composites (baseline from retrospective)
    ("binary_affinity_probability_binary", "binary_complex_iplddt"),
    ("binary_iptm", "binary_complex_iplddt"),
    ("binary_confidence_score", "binary_plddt_pocket"),
    # Ternary-only composites
    ("ternary_iptm", "ternary_complex_iplddt"),
    ("ternary_iptm", "ternary_plddt_pocket"),
    ("ternary_confidence_score", "ternary_plddt_pocket"),
    ("ternary_iptm", "ternary_plddt_hab1"),
    # Cross binary+ternary composites
    ("binary_iptm", "ternary_iptm"),
    ("binary_confidence_score", "ternary_iptm"),
    ("binary_complex_iplddt", "ternary_iptm"),
    ("binary_affinity_probability_binary", "ternary_iptm"),
    ("binary_plddt_pocket", "ternary_iptm"),
    ("binary_iptm", "ternary_confidence_score"),
    ("binary_affinity_probability_binary", "ternary_confidence_score"),
    ("binary_complex_iplddt", "ternary_plddt_pocket"),
    # Trp211 lock distance composites
    ("ternary_trp211_ligand_distance", "ternary_iptm"),
    ("ternary_trp211_ligand_distance", "ternary_plddt_ligand"),
    ("ternary_trp211_ligand_distance", "binary_plddt_pocket"),
    ("ternary_trp211_ligand_distance", "binary_affinity_probability_binary"),
    # ternary_complex_iplddt composites (best ternary metric from paired-MSA run)
    ("binary_affinity_probability_binary", "ternary_complex_iplddt"),
    ("binary_complex_iplddt", "ternary_complex_iplddt"),
    ("binary_plddt_pocket", "ternary_complex_iplddt"),
    ("binary_iptm", "ternary_complex_iplddt"),
    ("ternary_confidence_score", "ternary_complex_iplddt"),
    ("ternary_trp211_ligand_distance", "ternary_complex_iplddt"),
    # Pose consistency composites (lower RMSD = consistent = binder hypothesis)
    ("ligand_rmsd_binary_vs_ternary", "ternary_iptm"),
    ("ligand_rmsd_binary_vs_ternary", "ternary_complex_iplddt"),
    ("ligand_rmsd_binary_vs_ternary", "binary_affinity_probability_binary"),
    ("ligand_rmsd_binary_vs_ternary", "binary_complex_iplddt"),
    ("ligand_rmsd_binary_vs_ternary", "binary_plddt_pocket"),
    ("pocket_rmsd_binary_vs_ternary", "ternary_iptm"),
    ("pocket_rmsd_binary_vs_ternary", "ternary_complex_iplddt"),
    ("pocket_rmsd_binary_vs_ternary", "binary_affinity_probability_binary"),
    ("ligand_rmsd_binary_vs_ternary", "pocket_rmsd_binary_vs_ternary"),
    # hbond_distance_delta composites: high delta = ternary displaced ligand = binder
    # (binary correct pose disrupted by HAB1 Trp385 insertion in ternary)
    ("binary_plddt_pocket", "hbond_distance_delta"),
    ("binary_complex_iplddt", "hbond_distance_delta"),
    ("binary_affinity_probability_binary", "hbond_distance_delta"),
    ("ternary_complex_iplddt", "hbond_distance_delta"),
    ("binary_iptm", "hbond_distance_delta"),
    # ternary_hbond_distance composites (direction now fixed: higher = binder)
    ("binary_plddt_pocket", "ternary_hbond_distance"),
    ("binary_complex_iplddt", "ternary_hbond_distance"),
    ("binary_affinity_probability_binary", "ternary_hbond_distance"),
    # COO-to-R116 artifact filter: measures the specific binary-only artifact
    # (carboxylate salt-bridging R116, which is blocked by HAB1 in ternary).
    # Higher coo_to_r116_dist = COO is far from R116 = no artifact = potentially correct.
    # Does NOT assume OH-vs-COO orientation at the water network (both can be correct).
    ("binary_coo_to_r116_dist", "binary_plddt_pocket"),
    ("binary_coo_to_r116_dist", "binary_complex_iplddt"),
    ("binary_coo_to_r116_dist", "binary_affinity_probability_binary"),
    ("binary_coo_to_r116_dist", "hbond_distance_delta"),
]


def filter_59R(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out 59R variants (trivial negatives)."""
    sig_map = labels_df.set_index("pair_id")["variant_signature"].to_dict()
    df = df.copy()
    df["signature"] = df["name"].map(sig_map)
    mask = df["signature"].str.contains("59R", na=False) & (df["label"] == 0)
    return df[~mask].drop(columns=["signature"])


def run_pooled_ablation(loaded: dict) -> None:
    """Compare true pooled AUC for LCA-only, LCA+GLCA, and all three ligands.

    loaded: dict mapping ligand key -> (df_all, df_no59R)
    """
    configs = [
        ("LCA only",            ["lca"]),
        ("LCA + GLCA",          ["lca", "glca"]),
        ("LCA + GLCA + LCA-3-S", ["lca", "glca", "lca3s"]),
    ]

    print(f"\n\n{'='*70}")
    print(f"  TRAINING SET ABLATION (true pooled AUC on concatenated rows)")
    print(f"{'='*70}")
    print("  (Z-scores computed globally across the pooled dataset)")

    for filter_label, filter_key in [("All variants", "all"), ("No 59R negatives", "no59R")]:
        print(f"\n  == {filter_label} ==")
        for config_name, keys in configs:
            frames = [loaded[k][filter_key] for k in keys if k in loaded]
            if not frames:
                continue
            pooled_df = pd.concat(frames, ignore_index=True)
            n_pos = pooled_df["label"].sum()
            n_neg = len(pooled_df) - n_pos
            print(f"\n  --- {config_name}  (N={len(pooled_df)}, {n_pos} pos, {n_neg} neg) ---")

            aucs = compute_aucs(pooled_df, ALL_METRICS, HIGHER_IS_BETTER)
            comp_aucs = compute_composite_aucs(pooled_df, COMPOSITE_PAIRS, HIGHER_IS_BETTER)
            all_aucs = {**aucs, **comp_aucs}

            if all_aucs:
                print(f"\n  {'Metric':<55} {'AUC':>6}")
                print(f"  {'-'*55} {'-'*6}")
                for m, auc in sorted(all_aucs.items(), key=lambda x: -x[1])[:30]:
                    marker = " ***" if auc >= 0.70 else " **" if auc >= 0.65 else " *" if auc >= 0.60 else ""
                    print(f"  {m:<55} {auc:.4f}{marker}")


def main():
    all_results = []
    loaded_dfs: dict = {}  # lig_key -> {"all": df, "no59R": df}

    for lig_key, lig_name in LIGANDS.items():
        print(f"\n{'='*70}")
        print(f"  {lig_name} ({lig_key})")
        print(f"{'='*70}")

        try:
            df = load_merged_with_labels(lig_key)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        labels_df = pd.read_csv(LABELS_DIR / f"boltz_{lig_key}_binary.csv")

        n_pos = df["label"].sum()
        n_neg = len(df) - n_pos
        print(f"  Rows: {len(df)} ({n_pos} binders, {n_neg} non-binders)")

        # Check completeness
        has_binary = df["binary_iptm"].notna().sum()
        has_ternary = df["ternary_iptm"].notna().sum()
        print(f"  Binary results: {has_binary}/{len(df)}")
        print(f"  Ternary results: {has_ternary}/{len(df)}")

        loaded_dfs[lig_key] = {}
        for filter_name, filter_key, filter_fn in [
            ("All variants",     "all",   None),
            ("No 59R negatives", "no59R", lambda d: filter_59R(d, labels_df)),
        ]:
            sub = filter_fn(df) if filter_fn else df
            loaded_dfs[lig_key][filter_key] = sub
            n_pos_sub = sub["label"].sum()
            n_neg_sub = len(sub) - n_pos_sub

            print(f"\n  --- {filter_name} ({n_pos_sub} pos, {n_neg_sub} neg) ---")

            # Individual metric AUCs
            aucs = compute_aucs(sub, ALL_METRICS, HIGHER_IS_BETTER)
            if aucs:
                print(f"\n  {'Metric':<45} {'AUC':>6}")
                print(f"  {'-'*45} {'-'*6}")
                for m, auc in sorted(aucs.items(), key=lambda x: -x[1]):
                    marker = " ***" if auc >= 0.70 else " **" if auc >= 0.65 else " *" if auc >= 0.60 else ""
                    print(f"  {m:<45} {auc:.4f}{marker}")

            # Composite AUCs
            comp_aucs = compute_composite_aucs(sub, COMPOSITE_PAIRS, HIGHER_IS_BETTER)
            if comp_aucs:
                print(f"\n  {'Composite':<55} {'AUC':>6}")
                print(f"  {'-'*55} {'-'*6}")
                for m, auc in sorted(comp_aucs.items(), key=lambda x: -x[1]):
                    marker = " ***" if auc >= 0.70 else " **" if auc >= 0.65 else " *" if auc >= 0.60 else ""
                    print(f"  {m:<55} {auc:.4f}{marker}")

            # Store for mean-AUC pooled analysis
            for m, auc in aucs.items():
                all_results.append({"ligand": lig_key, "filter": filter_name, "metric": m, "auc": auc, "type": "single"})
            for m, auc in comp_aucs.items():
                all_results.append({"ligand": lig_key, "filter": filter_name, "metric": m, "auc": auc, "type": "composite"})

    # Pooled analysis
    print(f"\n\n{'='*70}")
    print(f"  POOLED ANALYSIS (mean AUC across ligands)")
    print(f"{'='*70}")

    results_df = pd.DataFrame(all_results)

    for filter_name in ["All variants", "No 59R negatives"]:
        sub = results_df[results_df["filter"] == filter_name]
        if sub.empty:
            continue

        print(f"\n  --- {filter_name} ---")
        pooled = sub.groupby("metric")["auc"].agg(["mean", "std", "count"])
        pooled = pooled[pooled["count"] == pooled["count"].max()]  # Only metrics present for all ligands
        pooled = pooled.sort_values("mean", ascending=False)

        print(f"\n  {'Metric':<55} {'Mean AUC':>9} {'Std':>6} {'N':>3}")
        print(f"  {'-'*55} {'-'*9} {'-'*6} {'-'*3}")
        for m, row in pooled.head(25).iterrows():
            marker = " ***" if row["mean"] >= 0.70 else " **" if row["mean"] >= 0.65 else " *" if row["mean"] >= 0.60 else ""
            print(f"  {m:<55} {row['mean']:.4f}   {row['std']:.4f} {int(row['count']):>3}{marker}")

    # Training set ablation: true pooled AUC on concatenated rows
    run_pooled_ablation(loaded_dfs)


if __name__ == "__main__":
    main()
