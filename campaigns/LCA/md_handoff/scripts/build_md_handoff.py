#!/usr/bin/env python3
"""
Build the MD-collaborator hand-off set for the LCA / LCA-3-S apo-MD experiment.

Deliverable = N_TOTAL (default 200) PYR1 designs, each with BOTH a LCA (LCAM) and
a LCA-3-S (LCA3S) Boltz holo structure:
  - ALL experimentally-labeled binders (LCA / LCA3S / both). 37 as of 2026-09-30:
    the original 35 plus the two prior clonally-validated positives POS_seq26
    (LCA + LCA-3-S) and POS_G0 (LCA only).
  - N_TOTAL - n_binders nonbinders (163 at 37 binders) DISTRIBUTION-MATCHED to the binders on the Boltz features that
    discriminate them (ligand pLDDT + water-network geometry + pocket pLDDT, per
    ligand), so a Boltz-only classifier is driven toward AUC ~0.5. Any binder/
    non-binder separation the MD then finds is signal Boltz did NOT already have.
    Pocket pLDDT is added because it is the strongest single Boltz discriminator
    (the motif proxy); leaving it un-matched would let Boltz keep that signal.

Matching = greedy nearest-neighbour (1:k, without replacement) in standardized
6-D feature space: [LCAM plddt_ligand, LCAM geometry_score, LCAM plddt_pocket,
                    LCA3S plddt_ligand, LCA3S geometry_score, LCA3S plddt_pocket].

Constitutive designs are deliberately excluded (set aside for now).
"""
import argparse, csv, os, shutil, glob
import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
# --- defaults point at the 5-SEED aggregated tables. The shipped single-seed
# --- hand-off (results/md_handoff/) is never overwritten by this script again.
SCRATCH = "/gpfs/alpine1/scratch/ryde3462/lca_lca3s_5seed"
MANIFEST = f"{ROOT}/inputs/sequences/cofolding_sequence_manifest.csv"
SLIM = f"{ROOT}/results/md_handoff_5seed/boltz_plddt_iptm_geometry_pucker_5seed.csv"
FULL = f"{ROOT}/results/md_handoff_5seed/boltz_full_metrics_5seed.csv"
OUT = f"{ROOT}/results/md_handoff_5seed"
PDBDIR = f"{OUT}/pdbs"

N_TOTAL = 200            # hand-off size; nonbinder count = N_TOTAL - n_binders
N_NONBINDERS = None      # derived in main() once the binders are counted
LIGS = ["LCAM", "LCA3S"]
FEATS = ["plddt_ligand", "geometry_score", "plddt_pocket"]  # per ligand -> 6 features
# Every feature whose residual is REPORTED, whether or not it is matched on.
# Dropping a feature from FEATS does NOT make it balanced -- it only stops the
# matcher pretending it is -- so the audit set is fixed and always prints.
AUDIT = ["plddt_ligand", "geometry_score", "plddt_pocket", "esm_apo_dg"]
BINDER_LABELS = {"LCA binder", "LCA3S binder", "both binder"}
NO_WRITE = False
SELECT_IDS = None
SEED = 0

# QC columns from aggregate_seeds.py, carried into the hand-off CSV so the
# collaborator (and any later filtering) can see how stable each pose was.
QC_PER_LIG = ["n_seeds", "rep_seed", "seed_agreement", "mode_majority",
              "flipped_fraction", "pose_rmsd_spread", "pose_rmsd_max",
              "head_clash_fraction", "r79_clash_fraction", "pass_pose_consistency"]


def parse_args():
    global SCRATCH, MANIFEST, SLIM, FULL, OUT, PDBDIR, N_TOTAL, FEATS, NO_WRITE
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--slim", default=SLIM)
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--n-total", type=int, default=N_TOTAL,
                    help="total hand-off size (binders + matched nonbinders)")
    ap.add_argument("--feats", default=",".join(FEATS),
                    help="comma-separated per-ligand features to MATCH on "
                         f"(default {','.join(FEATS)}); residuals for {','.join(AUDIT)} "
                         "are reported either way")
    ap.add_argument("--select-ids", default=None,
                    help="CSV with sequence_id[,class] naming a PRECOMPUTED selection "
                         "(e.g. from scripts/balance_select.py). Skips the built-in "
                         "distance matcher entirely and just writes/stages that set, so "
                         "there is one writer regardless of how the set was chosen.")
    ap.add_argument("--no-write", action="store_true",
                    help="report match quality only; write no CSVs and stage no PDBs "
                         "(use for sweeping --feats / --n-total)")
    a = ap.parse_args()
    SCRATCH, MANIFEST, SLIM, FULL = a.scratch, a.manifest, a.slim, a.full
    OUT, N_TOTAL, NO_WRITE = a.outdir, a.n_total, a.no_write
    global SELECT_IDS
    SELECT_IDS = a.select_ids
    FEATS = [f.strip() for f in a.feats.split(",") if f.strip()]
    PDBDIR = f"{OUT}/pdbs"
    if not NO_WRITE:
        os.makedirs(OUT, exist_ok=True)
        os.makedirs(PDBDIR, exist_ok=True)
    return a


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


def _cell(slim, full, sid, lig, f):
    """A feature may live in SLIM (Boltz) or only in FULL (e.g. esm_apo_dg)."""
    r = slim.get(sid, {}).get(lig, {})
    if f in r and r[f] != "":
        return r[f]
    fr = full.get((sid, lig), {}) or {}
    # FULL prefixes most Boltz columns with "binary_"; try both so a metric is
    # never silently SKIPPED just because it is absent from the SLIM table.
    for key in (f, f"binary_{f}"):
        if fr.get(key, "") != "":
            return fr[key]
    return ""


def feature_vec(slim, sid, feats=None, full=None):
    v = []
    for lig in LIGS:
        for f in (feats if feats is not None else FEATS):
            raw = _cell(slim, full or {}, sid, lig, f)
            v.append(float(raw) if raw not in ("", None) else np.nan)
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
    global N_NONBINDERS
    N_NONBINDERS = N_TOTAL - len(binders)
    print(f"binders={len(binders)}  nonbinder-pool={len(nonb)}")
    if not SELECT_IDS:   # with --select-ids the count comes from the file, not --n-total
        print(f"hand-off size {N_TOTAL} -> selecting {N_NONBINDERS} matched nonbinders")
    if N_NONBINDERS < 1:
        raise SystemExit(f"--n-total {N_TOTAL} is <= the binder count {len(binders)}")

    preset = None
    if SELECT_IDS:
        with open(SELECT_IDS) as fh:
            want = [r for r in csv.DictReader(fh)]
        ids = {r["sequence_id"] for r in want}
        preset = [s for s in nonb if s in ids]
        pb = [s for s in binders if s in ids]
        unknown = ids - set(nonb) - set(binders)
        print(f"--select-ids {SELECT_IDS}: {len(pb)} binders + {len(preset)} nonbinders"
              f"{f'  ({len(unknown)} ids not in the labeled pool, ignored)' if unknown else ''}")
        if len(pb) not in (0, len(binders)):
            print(f"  NOTE the file names {len(pb)} of {len(binders)} binders; "
                  "all binders are kept regardless")
        if not preset:
            raise SystemExit("--select-ids matched no nonbinders in the pool")
        N_NONBINDERS = len(preset)

    Xb = np.array([feature_vec(slim, s, full=full) for s in binders])
    Xn = np.array([feature_vec(slim, s, full=full) for s in nonb])
    if not (np.isfinite(Xb).all() and np.isfinite(Xn).all()):
        bad = [f"{l}_{f}" for l in LIGS for f in FEATS]
        miss = [bad[k] for k in range(len(bad))
                if not (np.isfinite(Xb[:, k]).all() and np.isfinite(Xn[:, k]).all())]
        raise SystemExit(
            f"matched feature(s) {miss} are blank/missing for some designs -- "
            "cannot match on a column that is not populated "
            "(esm_apo_dg needs scripts/join_esm_apo_dg.py to run first)")

    # standardize using pooled stats
    pool = np.vstack([Xb, Xn])
    mu, sd = pool.mean(0), pool.std(0) + 1e-9
    Zb = (Xb - mu) / sd
    Zn = (Xn - mu) / sd

    # distance matrix binders x nonbinders
    D = np.sqrt(((Zb[:, None, :] - Zn[None, :, :]) ** 2).sum(-1))  # (35, Npool)

    if preset is not None:
        # A precomputed selection skips the matcher; match_distance is then only a
        # reported diagnostic (distance to the nearest binder), not a selection key.
        idx = {s: j for j, s in enumerate(nonb)}
        sel_nb = list(preset)
        sel_nb_dist = {s: float(D[:, idx[s]].min()) for s in preset}
        print(f"using precomputed selection: {len(sel_nb)} nonbinders "
              "(match_distance = distance to nearest binder, diagnostic only)")
    else:
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
    # Audited over AUDIT, not FEATS: a feature dropped from the matcher still has
    # a residual, and hiding it is exactly how a set looks better blinded than it is.
    matched = {f"{l}_{f}" for l in LIGS for f in FEATS}
    anames = [f"{l}_{f}" for l in LIGS for f in AUDIT]
    Ab = np.array([feature_vec(slim, s, AUDIT, full) for s in binders])
    An = np.array([feature_vec(slim, s, AUDIT, full) for s in nonb])
    Asel = np.array([feature_vec(slim, s, AUDIT, full) for s in sel_nb])
    print(f"\nmatched on {len(matched)} features: {sorted(matched)}")
    print("AUC binder-vs-NB  (0.5 = indistinguishable; * = matched on):")
    print(f"  {'feature':24s} {'vs ALL-NB':>10s} {'vs MATCHED':>11s}")
    aucs, worst = {}, 0.0
    for k, name in enumerate(anames):
        pos, neg, sel = Ab[:, k], An[:, k], Asel[:, k]
        ok = np.isfinite(pos).all() and np.isfinite(neg).all() and np.isfinite(sel).all()
        if not ok:
            print(f"  {name:24s} {'--':>10s} {'--':>11s}   (not populated)")
            continue
        a_all, a_sel = auc(pos, neg), auc(pos, sel)
        aucs[name] = (a_all, a_sel)
        star = "*" if name in matched else " "
        print(f"  {name:24s} {a_all:10.3f} {a_sel:10.3f} {star}")
        if name in matched:
            worst = max(worst, abs(a_sel - 0.5))
    if worst:
        print(f"  worst |AUC-0.5| across MATCHED features: {worst:.3f}")

    if NO_WRITE:
        print("\n--no-write: no CSVs written, no PDBs staged")
        return binders, sel_nb, aucs

    # ---- write selection CSV ----
    extra_cols = ["binary_confidence_score", "binary_iptm", "binary_ligand_iptm",
                  "binary_affinity_probability_binary", "binary_hbond_distance",
                  "binary_hbond_angle", "binary_binding_mode", "rings_distorted",
                  "ring_pucker_max", "esm_apo_dg"]
    csv_path = f"{OUT}/md_handoff_selection.csv"
    with open(csv_path, "w", newline="") as fh:
        base = ["sequence_id", "class", "manifest_label", "pocket_sequence",
                "protein_sequence", "match_distance"]
        # plddt_pocket is a MATCHED feature now, so it has to be auditable in the
        # output, not just consumed by the matcher.
        slimcols = ["plddt_ligand", "geometry_score", "plddt_pocket",
                    "geometry_dist_score", "geometry_ang_score", "iptm",
                    "ligand_iptm", "hbond_distance", "hbond_angle"]
        percols = []
        for lig in LIGS:
            for c in slimcols:
                percols.append(f"{lig}_{c}")
            for c in extra_cols:
                percols.append(f"{lig}_{c}")
            for c in QC_PER_LIG:
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
                for c2 in slimcols:
                    row.append(s.get(c2, ""))
                fr = full.get((sid, lig), {})
                for c2 in extra_cols:
                    row.append(fr.get(c2, ""))
                for c2 in QC_PER_LIG:
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

    stage_pdbs(binders + sel_nb, full)
    return binders, sel_nb, aucs


def stage_pdbs(sids, full):
    """Copy each selected design's REPRESENTATIVE-SEED model_0 into OUT/pdbs.

    With a 5-seed ensemble there is no single "the" structure, so the staged PDB is
    the rep_seed chosen by aggregate_seeds.py (medoid of the majority-mode cluster).
    Falls back to any seed only if the rep_seed file is missing, and reports that.
    """
    # Wipe stale PDBs first. Re-running with a different selection used to leave the
    # previous run's structures behind, so pdbs/ held more designs than the selection
    # CSV listed (same failure mode as the stale build_farm.py symlinks).
    if os.path.isdir(PDBDIR):
        stale = [f for f in os.listdir(PDBDIR) if f.endswith(".pdb")]
        for f in stale:
            os.remove(f"{PDBDIR}/{f}")
        if stale:
            print(f"cleared {len(stale)} pre-existing PDBs from {PDBDIR}")
    else:
        os.makedirs(PDBDIR, exist_ok=True)

    n_ok = 0
    missing, fellback = [], []
    for sid in sids:
        for lig in LIGS:
            name = f"{sid}__{lig}_binary"
            dst = f"{PDBDIR}/{sid}__{lig}.pdb"
            rep = (full.get((sid, lig), {}) or {}).get("rep_seed", "")
            cands = []
            if rep:
                cands.append(f"{SCRATCH}/seed_{rep}/boltz_out/boltz_results_*/predictions/{name}/{name}_model_0.pdb")
            cands.append(f"{SCRATCH}/seed_*/boltz_out/boltz_results_*/predictions/{name}/{name}_model_0.pdb")
            src = None
            for k, pat in enumerate(cands):
                h = sorted(glob.glob(pat))
                if h:
                    src = h[0]
                    if k > 0:
                        fellback.append(f"{sid}__{lig}(rep_seed={rep or '?'})")
                    break
            if not src:
                missing.append(f"{sid}__{lig}")
                continue
            shutil.copyfile(src, dst)
            n_ok += 1
    print(f"\nstaged {n_ok} PDBs -> {PDBDIR}  (expected {2*len(sids)})")
    if fellback:
        print(f"  WARNING rep_seed PDB absent, used another seed for {len(fellback)}: {fellback[:5]}")
    if missing:
        print(f"  WARNING no PDB at all for {len(missing)}: {missing[:5]}")


if __name__ == "__main__":
    parse_args()
    main()
