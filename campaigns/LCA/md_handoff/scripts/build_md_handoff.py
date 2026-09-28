#!/usr/bin/env python3
"""
Build the MD-collaborator hand-off set for the LCA / LCA-3-S apo-MD experiment.

Deliverable = 200 PYR1 designs, each with BOTH a LCA (LCAM) and a LCA-3-S (LCA3S)
Boltz holo structure:
  - all 35 experimentally-labeled binders (LCA / LCA3S / both)
  - 165 nonbinders DISTRIBUTION-MATCHED to the binders on the Boltz features that
    discriminate them (ligand pLDDT + water-network geometry, per ligand), so a
    Boltz-only classifier is driven toward AUC ~0.5. Any binder/non-binder
    separation the MD then finds is signal Boltz did NOT already have.

Matching = greedy nearest-neighbour (1:k, without replacement) in standardized
4-D feature space: [LCAM plddt_ligand, LCAM geometry_score,
                    LCA3S plddt_ligand, LCA3S geometry_score].

Constitutive designs are deliberately excluded (set aside for now).
"""
import csv, os, shutil, glob
import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
SCRATCH = "/gpfs/alpine1/scratch/ryde3462/lca_lca3s_full/boltz_out"
MANIFEST = f"{ROOT}/inputs/sequences/cofolding_sequence_manifest.csv"
SLIM = f"{ROOT}/results/full_binary/boltz_plddt_iptm_geometry_pucker.csv"
FULL = f"{ROOT}/results/full_binary/boltz_full_metrics.csv"
OUT = f"{ROOT}/results/md_handoff"
PDBDIR = f"{OUT}/pdbs"

N_NONBINDERS = 165
LIGS = ["LCAM", "LCA3S"]
FEATS = ["plddt_ligand", "geometry_score"]  # per ligand -> 4 features
BINDER_LABELS = {"LCA binder", "LCA3S binder", "both binder"}
SEED = 0

os.makedirs(OUT, exist_ok=True)
os.makedirs(PDBDIR, exist_ok=True)


def load():
    mani = {r["sequence_id"]: r for r in csv.DictReader(open(MANIFEST))}
    slim = {}  # sid -> lig -> row
    for r in csv.DictReader(open(SLIM)):
        sid = r["name"].split("__")[0]
        slim.setdefault(sid, {})[r["ligand"]] = r
    full = {}  # (sid,lig) -> row  (extra metrics for the CSV)
    for r in csv.DictReader(open(FULL)):
        sid = r["name"].split("__")[0]
        full[(sid, r["ligand"])] = r
    return mani, slim, full


def klass(mani, sid):
    lab = mani[sid]["label"]
    if lab in BINDER_LABELS:
        return "binder"
    if lab == "nonbinder":
        return "nonbinder"
    return "other"  # constitutive etc -> excluded


def feature_vec(slim, sid):
    v = []
    for lig in LIGS:
        for f in FEATS:
            v.append(float(slim[sid][lig][f]))
    return v


def auc(pos, neg):
    """Mann-Whitney AUC: P(pos>neg)."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    a = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return a


def main():
    mani, slim, full = load()
    rng = np.random.default_rng(SEED)

    binders, nonb = [], []
    for sid in slim:
        if sid not in mani:
            continue
        if any(l not in slim[sid] for l in LIGS):
            continue
        c = klass(mani, sid)
        if c == "binder":
            binders.append(sid)
        elif c == "nonbinder":
            nonb.append(sid)
    binders.sort(); nonb.sort()
    print(f"binders={len(binders)}  nonbinder-pool={len(nonb)}")

    Xb = np.array([feature_vec(slim, s) for s in binders])
    Xn = np.array([feature_vec(slim, s) for s in nonb])

    # standardize using pooled stats
    pool = np.vstack([Xb, Xn])
    mu, sd = pool.mean(0), pool.std(0) + 1e-9
    Zb = (Xb - mu) / sd
    Zn = (Xn - mu) / sd

    # distance matrix binders x nonbinders
    D = np.sqrt(((Zb[:, None, :] - Zn[None, :, :]) ** 2).sum(-1))  # (35, Npool)

    # greedy 1:k round-robin without replacement
    taken = np.zeros(len(nonb), dtype=bool)
    picks = []  # (nb_index, matched_binder_index, dist)
    while len(picks) < N_NONBINDERS:
        remaining = N_NONBINDERS - len(picks)
        # this round: nearest available nonbinder for each binder
        round_choices = []
        for bi in range(len(binders)):
            avail = np.where(~taken)[0]
            if len(avail) == 0:
                break
            j = avail[D[bi, avail].argmin()]
            round_choices.append((D[bi, j], bi, j))
        if not round_choices:
            break
        # if the round would overshoot, keep the closest `remaining` matches
        round_choices.sort()  # by distance
        if len(round_choices) > remaining:
            round_choices = round_choices[:remaining]
        for dist, bi, j in round_choices:
            if taken[j]:
                continue  # a closer binder grabbed it this round
            taken[j] = True
            picks.append((j, bi, dist))

    sel_nb = [nonb[j] for (j, _, _) in picks][:N_NONBINDERS]
    sel_nb_dist = {nonb[j]: d for (j, _, d) in picks}
    print(f"selected nonbinders={len(sel_nb)}  (target {N_NONBINDERS})")

    # ---- verify match quality: per-feature AUC binder-vs-selected-NB ----
    Xsel = np.array([feature_vec(slim, s) for s in sel_nb])
    Xall = Xn
    fnames = [f"{l}_{f}" for l in LIGS for f in FEATS]
    print("\nAUC binder-vs-NB  (0.5 = indistinguishable):")
    print(f"  {'feature':22s} {'vs ALL-NB':>10s} {'vs MATCHED':>10s}")
    aucs = {}
    for k, name in enumerate(fnames):
        a_all = auc(Xb[:, k], Xall[:, k])
        a_sel = auc(Xb[:, k], Xsel[:, k])
        aucs[name] = (a_all, a_sel)
        print(f"  {name:22s} {a_all:10.3f} {a_sel:10.3f}")

    # ---- write selection CSV ----
    extra_cols = ["binary_confidence_score", "binary_iptm", "binary_ligand_iptm",
                  "binary_affinity_probability_binary", "binary_hbond_distance",
                  "binary_hbond_angle", "binary_binding_mode", "rings_distorted",
                  "ring_pucker_max", "esm_apo_dg"]
    csv_path = f"{OUT}/md_handoff_selection.csv"
    with open(csv_path, "w", newline="") as fh:
        base = ["sequence_id", "class", "manifest_label", "pocket_sequence",
                "protein_sequence", "match_distance"]
        percols = []
        for lig in LIGS:
            for c in ["plddt_ligand", "geometry_score", "geometry_dist_score",
                      "geometry_ang_score", "iptm", "ligand_iptm",
                      "hbond_distance", "hbond_angle"]:
                percols.append(f"{lig}_{c}")
            for c in extra_cols:
                percols.append(f"{lig}_{c}")
            percols.append(f"{lig}_pdb")
        w = csv.writer(fh); w.writerow(base + percols)
        for sid in binders + sel_nb:
            c = "binder" if sid in binders else "nonbinder"
            row = [sid, c, mani[sid]["label"], mani[sid]["pocket_sequence"],
                   mani[sid]["protein_sequence"],
                   f"{sel_nb_dist.get(sid, 0.0):.4f}"]
            for lig in LIGS:
                s = slim[sid][lig]
                for c2 in ["plddt_ligand", "geometry_score", "geometry_dist_score",
                           "geometry_ang_score", "iptm", "ligand_iptm",
                           "hbond_distance", "hbond_angle"]:
                    row.append(s.get(c2, ""))
                fr = full.get((sid, lig), {})
                for c2 in extra_cols:
                    row.append(fr.get(c2, ""))
                row.append(f"pdbs/{sid}__{lig}.pdb")
            w.writerow(row)
    print(f"\nwrote {csv_path}  ({len(binders)+len(sel_nb)} sequences)")

    # ---- figure-ready long-format (one row per sequence x ligand) ----
    long_path = f"{OUT}/md_handoff_figure_data_long.csv"
    with open(long_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence_id", "class", "in_selection", "ligand",
                    "plddt_ligand", "geometry_score"])
        sel_set = set(binders) | set(sel_nb)
        for sid in slim:
            if sid not in mani:
                continue
            c = klass(mani, sid)
            if c == "other":
                continue
            for lig in LIGS:
                if lig not in slim[sid]:
                    continue
                s = slim[sid][lig]
                w.writerow([sid, c, int(sid in sel_set), lig,
                            s["plddt_ligand"], s["geometry_score"]])
    print(f"wrote {long_path}")

    return binders, sel_nb, aucs


if __name__ == "__main__":
    main()
