#!/usr/bin/env python3
"""Merge ligand_geometry_scan.py shards and evaluate ligand geometry two ways.

1. AS A METRIC. Per (design, ligand): the fraction of the 5 seeds that are
   geometry-valid (correct stereo at every centre AND no collapsed atoms), and the
   stereo-only fraction. Reported as binder / non-binder / constitutive AUCs over
   the FULL pool, plus Spearman against the co-folding scores balance_select.py
   balances -- if it is orthogonal to those, it is information the balanced MD set
   still leaks.
2. AS A FEASIBILITY MAP. Which designs have >=1 geometry-valid seed per ligand,
   i.e. which designs a stereo-correct representative can be drawn from.

Nothing is dropped here. Outputs:
  ligand_geometry_long.csv     one row per structure (merged shards)
  ligand_geometry_by_design.csv one row per (design, ligand)
"""
import argparse, glob
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
BALANCED = ["plddt_ligand", "plddt_pocket", "plddt_protein", "geometry_score",
            "hbond_distance", "iptm", "affinity_probability_binary",
            "n_interface_unsatisfied"]


def auc(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    return u.statistic / (len(x) * len(y)), u.pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{ROOT}/results/ligand_geometry_scan")
    ap.add_argument("--manifest-csv", default=f"{ROOT}/inputs/sequences/cofolding_sequence_manifest.csv")
    ap.add_argument("--full", nargs="+", default=[
        f"{ROOT}/results/md_handoff_5seed/boltz_full_metrics_5seed.csv",
        f"{ROOT}/results/constitutive_5seed/boltz_full_metrics_5seed.csv"])
    ap.add_argument("--expect", type=int, default=16740)
    a = ap.parse_args()

    shards = sorted(glob.glob(f"{a.dir}/shards/shard_*.csv"))
    L = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    print(f"merged {len(shards)} shards -> {len(L)} rows (expect {a.expect})")
    if L.path.duplicated().any():
        raise SystemExit(f"DUPLICATE paths: {L.path.duplicated().sum()}")
    if len(L) != a.expect:
        print(f"  WARNING row count {len(L)} != {a.expect}")
    bad_eval = (L.geom_evaluable != 1).sum()
    print(f"structures NOT evaluable: {bad_eval}  (these count as invalid, never as a pass)")
    L.to_csv(f"{a.dir}/ligand_geometry_long.csv", index=False)

    man = pd.read_csv(a.manifest_csv)
    idc = [c for c in man.columns if "id" in c.lower()][0]
    lab = dict(zip(man[idc].astype(str), man["label"]))
    L["label"] = L.sequence_id.astype(str).map(lab)
    L["cls"] = np.where(L.label.astype(str).str.contains("binder")
                        & ~L.label.astype(str).str.contains("non"), "binder", L.label)

    g = (L.groupby(["sequence_id", "ligand", "cls"])
          .agg(n_seeds=("geom_ok", "size"),
               geom_ok_frac=("geom_ok", "mean"),
               stereo_ok_frac=("stereo_ok", "mean"),
               n_valid_seeds=("geom_ok", "sum"),
               c3_inverted_frac=("inv_C3", "mean"),
               c5_inverted_frac=("inv_C5", "mean"),
               collapsed_frac=("collapsed", "mean"))
          .reset_index())
    g.to_csv(f"{a.dir}/ligand_geometry_by_design.csv", index=False)

    print("\n=== structure-level rates (all 5 seeds pooled) ===")
    print(L.groupby(["ligand", "cls"]).agg(
        n=("geom_ok", "size"), geom_ok=("geom_ok", "mean"),
        C3_inv=("inv_C3", "mean"), C5_inv=("inv_C5", "mean"),
        C20_inv=("inv_C20", "mean"), collapsed=("collapsed", "mean")).round(3).to_string())

    print("\n=== AS A METRIC: geom_ok_frac (fraction of seeds geometry-valid) ===")
    for metric in ("geom_ok_frac", "stereo_ok_frac"):
        print(f"\n  {metric}")
        for lg in ("LCAM", "LCA3S"):
            s = g[g.ligand == lg]
            c = {k: s[s.cls == k][metric] for k in ("binder", "nonbinder", "constitutive")}
            a1, p1 = auc(c["binder"], c["nonbinder"])
            a2, p2 = auc(c["constitutive"], c["nonbinder"])
            a3, p3 = auc(c["constitutive"], c["binder"])
            print(f"    {lg:6} binder-vs-NB {a1:.3f} (p={p1:.2g})   const-vs-NB {a2:.3f} "
                  f"(p={p2:.2g})   const-vs-binder {a3:.3f} (p={p3:.2g})")

    full = pd.concat([pd.read_csv(f) for f in a.full], ignore_index=True)
    full["sequence_id"] = full["name"].astype(str).str.replace(
        r"__(LCAM|LCA3S)_binary$", "", regex=True)
    print("\n=== orthogonality to the balanced co-folding scores (Spearman, full pool) ===")
    for lg in ("LCAM", "LCA3S"):
        s = g[g.ligand == lg].merge(full[full.ligand == lg], on=["sequence_id", "ligand"])
        out = []
        for f in BALANCED:
            col = f if f in s.columns else f"binary_{f}"
            if col not in s.columns:
                continue
            v = pd.to_numeric(s[col], errors="coerce")
            m = v.notna()
            r = stats.spearmanr(s.geom_ok_frac[m], v[m])
            out.append(f"{f}={r.correlation:+.2f}")
        print(f"  {lg} (n={len(s)}): " + "  ".join(out))

    print("\n=== AS A FEASIBILITY MAP: designs with >=1 geometry-valid seed ===")
    for lg in ("LCAM", "LCA3S"):
        s = g[g.ligand == lg]
        print(f"  {lg}:")
        for k in ("binder", "nonbinder", "constitutive"):
            t = s[s.cls == k]
            print(f"    {k:13} {int((t.n_valid_seeds > 0).sum()):5d} / {len(t):5d} "
                  f"({100 * (t.n_valid_seeds > 0).mean():5.1f}%)")
    w = g.pivot_table(index=["sequence_id", "cls"], columns="ligand",
                      values="n_valid_seeds").reset_index()
    w["both_ok"] = (w.LCAM > 0) & (w.LCA3S > 0)
    print("\n  designs with a valid seed for BOTH ligands:")
    for k in ("binder", "nonbinder", "constitutive"):
        t = w[w.cls == k]
        print(f"    {k:13} {int(t.both_ok.sum()):5d} / {len(t):5d} ({100 * t.both_ok.mean():5.1f}%)")
    print(f"\nwrote {a.dir}/ligand_geometry_long.csv and ligand_geometry_by_design.csv")


if __name__ == "__main__":
    main()
