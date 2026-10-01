#!/usr/bin/env python3
"""
Multi-score balanced selection of matched nonbinders.

WHY THIS EXISTS
build_md_handoff.py matches by greedy nearest-neighbour on Euclidean distance in
standardized feature space. That optimizes the WRONG objective: distance, not
balance. With six features of very unequal separability, equal-weighting them
leaves the dominant one badly unbalanced -- LCA pocket pLDDT lands at AUC 0.739
while LCA3S geometry lands at 0.421. Collapsing to a single feature fixes that
one score and lets the others leak. Neither is multi-score balance.

THE REFORMULATION
The binder set is FIXED, so for feature f and candidate nonbinder j:

    AUC_f(S) = P(binder > nonbinder) = (1/|S|) * sum_{j in S} c_f[j]
    c_f[j]   = ( #{i: b_i > x_j} + 0.5*#{i: b_i == x_j} ) / n_binders

Each candidate's contribution to each AUC is a CONSTANT, precomputed once. So
"drive every score to 0.5" becomes: choose S, |S| = k, minimizing

    max_f | mean_{j in S} c_f[j] - 0.5 |

which is a linear subset-balancing problem -- solvable directly (greedy seed +
local swaps) rather than approached sideways via distance. It also makes the
pool limit exact rather than empirical: see --report-bounds.

CAVEAT, STATE IT OUT LOUD
This balances MARGINAL AUCs, one score at a time. A multivariate classifier on
the same features can still separate the groups even when every marginal sits at
0.5. Balanced marginals is what "Boltz-only AUC ~= 0.5 per score" means, not
"no Boltz information remains".

PER-ARM SELECTION (2026-10-01)
The ligand-geometry gate kills single arms, not whole designs: a design can have a
usable LCA structure and no usable LCA-3-S structure. Requiring pairs would discard
6 good binder arms, and the collaborator does not need paired structures. So each
ligand arm is balanced as its OWN subset-selection problem (8 scores -> 80
constraints each) and the output is a list of (design, ligand) arms. A design may
appear in one arm or both.

Arm labels are PER LIGAND: a design labelled "LCA binder" is a POSITIVE in the LCA
arm and a NEGATIVE in the LCA-3-S arm, because that is what the EC50 data says.

  python3 scripts/balance_select.py --n-per-arm 137
  python3 scripts/balance_select.py --n-per-arm 137 --report-bounds
  python3 scripts/balance_select.py --n-per-arm 137 --write   # emit the selection
"""
import argparse, csv, os, sys
import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
BASE = f"{ROOT}/results/md_handoff_5seed"
LIGS = ["LCAM", "LCA3S"]
BINDER_LABELS = {"LCA binder", "LCA3S binder", "both binder"}
CORE_FEATS = ["plddt_ligand", "plddt_pocket", "plddt_protein",
              "iptm", "affinity_probability_binary", "n_interface_unsatisfied",
              # The geometry gate is itself mildly class-informative (binders keep more
              # valid seeds, AUC ~0.63), so how many seeds survived it is balanced too --
              # otherwise "how often Boltz got the stereochemistry right" stays available
              # to a classifier as a proxy for binding. Costs nothing: same max k.
              "n_valid_seeds"]
# geometry_score / hbond_distance measure the distance from the 3QN1 conserved-water
# position to the ligand's nearest O/N, scored as a Gaussian centred on 2.7 A -- i.e.
# they assume the water is RETAINED and the ligand H-bonds to it.
#
#   LCA  : bare 3alpha-OH, water retained, OH H-bonds to it. Binders 1.96 A vs
#          nonbinders 2.59 A, AUC 0.690. The score means what it says -> BALANCE IT.
#   LCA-3-S: the C3 sulfate oxygens sit 0.37-1.6 A from the water site, i.e. the
#          sulfate OCCUPIES the water position and does the water's job. The 2.7 A
#          ideal is then meaningless and the score inverts (binders 0.096 vs
#          nonbinders 0.382, AUC 0.124). Balancing a meaningless score would throw
#          away real structures to equalise an artefact -> DROP IT, and state the
#          sulfate-displaces-water hypothesis for MD to test instead.
ARM_FEATS = {
    "LCAM":  CORE_FEATS + ["geometry_score", "hbond_distance"],
    "LCA3S": CORE_FEATS,
}
# Reported but NOT balanced, so residual leakage is visible rather than assumed away.
DIAGNOSTIC_FEATS = ["geometry_score", "hbond_distance",
                    "n_valid_seeds", "geom_valid_fraction", "stereo_ok_fraction"]
DEFAULT_FEATS = ""      # empty -> use ARM_FEATS per ligand

# Per-ligand truth: a design labelled "LCA binder" binds LCA and NOT LCA-3-S, so its
# LCA-3-S arm is a NEGATIVE. Settled with the user 2026-10-01.
BINDS = {
    "both binder":  {"LCAM": 1, "LCA3S": 1},
    "LCA binder":   {"LCAM": 1, "LCA3S": 0},
    "LCA3S binder": {"LCAM": 0, "LCA3S": 1},
    "nonbinder":    {"LCAM": 0, "LCA3S": 0},
}


def arm_class(label, lig):
    """-> 'binder' / 'nonbinder' / None (None = not part of the labelled pool)."""
    if label not in BINDS:
        return None                      # constitutive, or anything unlabelled
    return "binder" if BINDS[label][lig] else "nonbinder"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{ROOT}/inputs/sequences/cofolding_sequence_manifest.csv")
    ap.add_argument("--slim", default=f"{BASE}/boltz_plddt_iptm_geometry_pucker_5seed.csv")
    ap.add_argument("--full", default=f"{BASE}/boltz_full_metrics_5seed.csv")
    ap.add_argument("--n-per-arm", default="111,117",
                    help="structures PER LIGAND ARM (binders + matched nonbinders), either "
                         "one number for every arm or one per ligand in LCAM,LCA3S order. "
                         "Selection is per (design, ligand) arm since the geometry gate "
                         "kills single arms, so there is no single --n-total any more. The "
                         "defaults are the largest fully-feasible size for each arm's own "
                         "feature set (--report-bounds proves it).")
    ap.add_argument("--feats", default=DEFAULT_FEATS,
                    help="features to BALANCE (comma-separated), applied to EVERY arm. "
                         "Empty (the default) uses ARM_FEATS, which balances the water-"
                         "network scores in the LCA arm and not in the LCA-3-S arm.")
    ap.add_argument("--mode", choices=["auc", "cdf", "both"], default="both",
                    help="auc = match mean rank only (AUC->0.5); cdf = match the whole "
                         "distribution on a quantile grid; both = default. AUC alone "
                         "leaves shape free: a set can sit at AUC 0.50 and still fail a "
                         "KS test, which is what 'identical distributions' has to survive.")
    ap.add_argument("--grid", type=int, default=9,
                    help="interior quantile points per feature for --mode cdf/both")
    ap.add_argument("--max-hamming", type=int, default=None,
                    help="restrict the candidate pool to nonbinders within this pocket "
                         "Hamming distance of SOME binder. Metric balance and sequence "
                         "proximity are orthogonal: balancing alone gives a set no closer "
                         "in sequence than random, so if the point is 'non-binders near the "
                         "binding clusters' this has to be asked for explicitly.")
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--passes", type=int, default=40, help="max local-swap passes per restart")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-bounds", action="store_true",
                    help="print the exact achievable range of each score for this k")
    ap.add_argument("--out", default=f"{BASE}/balanced_selection_ids.csv")
    ap.add_argument("--write", action="store_true", help="write the selected ids to --out")
    return ap.parse_args()


def load(a):
    mani = {r["sequence_id"]: r for r in csv.DictReader(open(a.manifest))}
    slim = {}
    for r in csv.DictReader(open(a.slim)):
        slim.setdefault(r["name"].split("__")[0], {})[r["ligand"]] = r
    full = {}
    for r in csv.DictReader(open(a.full)):
        full[(r["name"].split("__")[0], r["ligand"])] = r
    return mani, slim, full


def cell(slim, full, sid, lig, f):
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


def n_valid(full, sid, lig):
    """Geometry-valid seeds for this arm. 0 = no usable structure exists -> excluded."""
    v = (full.get((sid, lig)) or {}).get("n_valid_seeds", "")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0        # absent gate column -> treat as unusable, never as a free pass


def build(a, mani, slim, full, feats, lig):
    """Constraint matrix for ONE ligand arm.

    Selection is per (design, ligand) arm, not per design: an arm is excluded when it
    has no geometry-valid seed, so a design may enter one arm, both, or neither. That
    unpairs the set by design -- the collaborator does not need paired structures and
    requiring them would discard 6 binders whose LCA arm is perfectly good.
    """
    binders, nonb, dropped = [], [], {"binder": 0, "nonbinder": 0}
    for sid in slim:
        if sid not in mani or lig not in slim[sid]:
            continue
        cls = arm_class(mani[sid]["label"], lig)
        if cls is None:
            continue
        if n_valid(full, sid, lig) < 1:
            dropped[cls] += 1
            continue
        (binders if cls == "binder" else nonb).append(sid)
    binders.sort(); nonb.sort()
    print(f"[{lig}] eligible: {len(binders)} binder arms, {len(nonb)} nonbinder arms "
          f"(dropped for no valid geometry: {dropped['binder']} binder, "
          f"{dropped['nonbinder']} nonbinder)")

    if a.max_hamming is not None:
        bp = [mani[s]["pocket_sequence"] for s in binders]
        keep = []
        for s in nonb:
            p = mani[s]["pocket_sequence"]
            if min(sum(x != y for x, y in zip(p, q)) for q in bp) <= a.max_hamming:
                keep.append(s)
        print(f"--max-hamming {a.max_hamming}: pool {len(nonb)} -> {len(keep)}")
        nonb = keep

    names, usable, diag = [], [], []
    # a score that IS balanced for this arm must not also be reported as a diagnostic
    for f in feats + [d for d in DIAGNOSTIC_FEATS if d not in feats]:
        nm = f"{lig}_{f}"
        b = np.array([_num(cell(slim, full, s, lig, f)) for s in binders])
        n = np.array([_num(cell(slim, full, s, lig, f)) for s in nonb])
        if not (np.isfinite(b).all() and np.isfinite(n).all()):
            if f in feats:
                print(f"  SKIP {nm}: not populated for all designs", file=sys.stderr)
            continue
        if f in feats:
            names.append(nm); usable.append((b, n))
        else:
            diag.append((nm, b, n))
    if not usable:
        sys.exit(f"no usable features for {lig}")

    # Every constraint is a subset MEAN, so it fits one matrix with per-column targets.
    #   AUC column : c[j] = (#{b > x_j} + 0.5*#{ties}) / n_b      -> target 0.5
    #   CDF column : c[j] = 1[x_j <= x_t]                          -> target F_binder(x_t)
    cols, tgt, cnames = [], [], []
    for (b, n), nm in zip(usable, names):
        if a.mode in ("auc", "both"):
            gt = (b[None, :] > n[:, None]).sum(1)
            eq = (b[None, :] == n[:, None]).sum(1)
            cols.append((gt + 0.5 * eq) / len(b)); tgt.append(0.5)
            cnames.append(f"{nm}|AUC")
        if a.mode in ("cdf", "both"):
            qs = np.linspace(0, 100, a.grid + 2)[1:-1]
            for q, x in zip(qs, np.percentile(b, qs)):
                cols.append((n <= x).astype(float))
                tgt.append(float((b <= x).mean()))
                cnames.append(f"{nm}|F(q{q:.0f})")
    return (binders, nonb, names, np.column_stack(cols), np.array(tgt), cnames, diag)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def obj(mean_c, tgt):
    """max |achieved - target| across constraints; L2 as tiebreak."""
    d = np.abs(mean_c - tgt)
    return d.max(), float((d ** 2).sum())


def optimize(C, tgt, k, restarts, passes, seed):
    n, m = C.shape
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        # greedy seed: add the candidate that most reduces the objective
        sel = np.zeros(n, dtype=bool)
        tot = np.zeros(m)
        if r == 0:
            order_pool = np.arange(n)
        else:
            order_pool = rng.permutation(n)
        for t in range(k):
            cand = order_pool[~sel[order_pool]]
            if r > 0 and t < k // 4:          # randomized warm start for diversity
                j = cand[rng.integers(len(cand))]
            else:
                sub = cand if len(cand) <= 4000 else rng.choice(cand, 4000, replace=False)
                cost = np.abs((tot + C[sub]) / (t + 1) - tgt).max(1)
                j = sub[int(cost.argmin())]
            sel[j] = True; tot += C[j]

        cur = obj(tot / k, tgt)
        # local swaps: exchange one selected for one unselected
        for _ in range(passes):
            improved = False
            si = np.where(sel)[0]
            ui = np.where(~sel)[0]
            # delta of the mean for every (out, in) pair, evaluated per selected row
            for o in si:
                base = tot - C[o]
                cost = np.abs((base + C[ui]) / k - tgt)
                mx = cost.max(1); l2 = (cost ** 2).sum(1)
                p = int(np.lexsort((l2, mx))[0])
                cand = (mx[p], float(l2[p]))
                if cand < cur:
                    j = ui[p]
                    sel[o] = False; sel[j] = True
                    tot = base + C[j]
                    cur = cand
                    improved = True
                    ui = np.where(~sel)[0]
            if not improved:
                break
        if best is None or cur < best[0]:
            best = (cur, sel.copy(), tot / k)
    return best


def bounds(C, k):
    """Exact best/worst achievable mean for each constraint at this k (that one alone).

    If the target lies outside [lo, hi] the constraint is UNSATISFIABLE at this k --
    a proof about the pool, not an artefact of the search.
    """
    S = np.sort(C, axis=0)
    return S[:k].mean(0), S[-k:].mean(0)


def run_arm(a, mani, slim, full, feats, lig, n_arm):
    """Balance ONE ligand arm. -> (binders, selected_nonbinders)."""
    print(f"\n{'=' * 72}\n{lig} arm   (balancing {len(feats)}: {','.join(feats)})\n{'=' * 72}")
    binders, nonb, names, C, tgt, cnames, diag = build(a, mani, slim, full, feats, lig)
    k = n_arm - len(binders)
    print(f"selecting k={k} nonbinder arms from {len(nonb)} "
          f"(arm total {n_arm} = {len(binders)} binders + {k})")
    if k > len(nonb):
        sys.exit(f"[{lig}] cannot select k={k} from a pool of {len(nonb)} "
                 f"(--max-hamming too tight, or --n-per-arm too large)")
    if k < 1:
        sys.exit(f"[{lig}] --n-per-arm {n_arm} is <= the binder count {len(binders)}")
    print(f"mode={a.mode}: {len(names)} scores -> {len(cnames)} constraints")

    if a.report_bounds:
        lo, hi = bounds(C, k)
        bad = [(nm, l, h, t) for nm, l, h, t in zip(cnames, lo, hi, tgt)
               if not (l <= t <= h)]
        print(f"\nreachability at k={k}: {len(bad)}/{len(cnames)} constraints UNSATISFIABLE")
        for nm, l, h, t in bad:
            print(f"  {nm:30s} target {t:.3f} outside [{l:.3f}, {h:.3f}]")
        if not bad:
            print("  (all targets reachable -- the pool does not forbid the claim)")

    (mx, l2), sel, mean_c = optimize(C, tgt, k, a.restarts, a.passes, a.seed)
    allc = C.mean(0)
    print(f"\n  {'score':30s} {'vs ALL-NB':>10s} {'BALANCED':>10s}")
    for i, nm in enumerate(cnames):
        if nm.endswith("|AUC"):
            print(f"    {nm[:-4]:30s} {allc[i]:10.3f} {mean_c[i]:10.3f}")
    d = np.abs(mean_c - tgt)
    print(f"  worst |achieved-target| over ALL {len(cnames)} constraints = {mx:.4f}"
          f"  (at {cnames[int(d.argmax())]};  sum sq = {l2:.5f})")

    ids = [nonb[j] for j in np.where(sel)[0]]
    # Diagnostics are NOT balanced, so their AUC in the selected set is the residual
    # leak. The geometry gate is itself mildly class-informative, so this is the
    # number that says how much of that survives into the deliverable.
    if diag:
        keep = set(ids)
        print(f"\n  NOT balanced (residual leak, AUC vs the SELECTED nonbinders):")
        for nm, b, n in diag:
            nsel = np.array([v for s, v in zip(nonb, n) if s in keep])
            if len(nsel) < 2:
                continue
            gt = (b[None, :] > nsel[:, None]).sum(1)
            eq = (b[None, :] == nsel[:, None]).sum(1)
            print(f"    {nm:30s} {((gt + 0.5 * eq) / len(b)).mean():10.3f}")
    return binders, ids


def main():
    a = parse_args()
    override = [f.strip() for f in a.feats.split(",") if f.strip()]
    sizes = [int(x) for x in str(a.n_per_arm).split(",")]
    if len(sizes) == 1:
        sizes *= len(LIGS)
    if len(sizes) != len(LIGS):
        sys.exit(f"--n-per-arm needs 1 or {len(LIGS)} values, got {len(sizes)}")
    mani, slim, full = load(a)

    picked, used_feats = {}, {}
    for lig, n_arm in zip(LIGS, sizes):
        used_feats[lig] = override or ARM_FEATS[lig]
        picked[lig] = run_arm(a, mani, slim, full, used_feats[lig], lig, n_arm)

    rows = []
    for lig in LIGS:
        binders, ids = picked[lig]
        rows += [(s, lig, "binder") for s in binders]
        rows += [(s, lig, "nonbinder") for s in ids]
    sids = {s for s, _, _ in rows}
    print(f"\n{'=' * 72}")
    print(f"TOTAL {len(rows)} arms = {len(rows)} structures over {len(sids)} distinct designs")
    for lig in LIGS:
        b, i = picked[lig]
        print(f"  {lig:6s} {len(b):3d} binder + {len(i):4d} nonbinder = {len(b) + len(i)} arms"
              f"   balanced on {len(used_feats[lig])} scores")
    both = sum(1 for s in sids
               if all(any(r[0] == s and r[1] == l for r in rows) for l in LIGS))
    print(f"  designs appearing in BOTH arms: {both}  (one arm only: {len(sids) - both})")

    if a.write:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sequence_id", "ligand", "class"])
            for r in sorted(rows):
                w.writerow(r)
        print(f"\nwrote {a.out}  ({len(rows)} arms)")
    else:
        print("\n(--write to emit the selection)")


if __name__ == "__main__":
    main()
