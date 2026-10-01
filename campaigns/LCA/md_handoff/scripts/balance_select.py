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

  python3 scripts/balance_select.py --n-total 200
  python3 scripts/balance_select.py --n-total 200 --feats plddt_ligand,geometry_score,plddt_pocket,esm_apo_dg
  python3 scripts/balance_select.py --n-total 200 --write   # emit the selection
"""
import argparse, csv, os, sys
import numpy as np

ROOT = "/projects/ryde3462/lca_lca3s_modelling"
BASE = f"{ROOT}/results/md_handoff_5seed"
LIGS = ["LCAM", "LCA3S"]
BINDER_LABELS = {"LCA binder", "LCA3S binder", "both binder"}
DEFAULT_FEATS = "plddt_ligand,geometry_score,plddt_pocket"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{ROOT}/inputs/sequences/cofolding_sequence_manifest.csv")
    ap.add_argument("--slim", default=f"{BASE}/boltz_plddt_iptm_geometry_pucker_5seed.csv")
    ap.add_argument("--full", default=f"{BASE}/boltz_full_metrics_5seed.csv")
    ap.add_argument("--n-total", type=int, default=200)
    ap.add_argument("--feats", default=DEFAULT_FEATS,
                    help="per-ligand features to BALANCE (comma-separated)")
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


def build(a, mani, slim, full, feats):
    binders, nonb = [], []
    for sid in slim:
        if sid not in mani or any(l not in slim[sid] for l in LIGS):
            continue
        lab = mani[sid]["label"]
        if lab in BINDER_LABELS:
            binders.append(sid)
        elif lab == "nonbinder":
            nonb.append(sid)
    binders.sort(); nonb.sort()

    if a.max_hamming is not None:
        bp = [mani[s]["pocket_sequence"] for s in binders]
        keep = []
        for s in nonb:
            p = mani[s]["pocket_sequence"]
            if min(sum(x != y for x, y in zip(p, q)) for q in bp) <= a.max_hamming:
                keep.append(s)
        print(f"--max-hamming {a.max_hamming}: pool {len(nonb)} -> {len(keep)}")
        nonb = keep

    names, usable = [], []
    for lig in LIGS:
        for f in feats:
            nm = f"{lig}_{f}"
            b = np.array([_num(cell(slim, full, s, lig, f)) for s in binders])
            n = np.array([_num(cell(slim, full, s, lig, f)) for s in nonb])
            if not (np.isfinite(b).all() and np.isfinite(n).all()):
                print(f"  SKIP {nm}: not populated for all designs", file=sys.stderr)
                continue
            names.append(nm); usable.append((b, n))
    if not usable:
        sys.exit("no usable features")

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
    return binders, nonb, names, np.column_stack(cols), np.array(tgt), cnames


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


def main():
    a = parse_args()
    feats = [f.strip() for f in a.feats.split(",") if f.strip()]
    mani, slim, full = load(a)
    binders, nonb, names, C, tgt, cnames = build(a, mani, slim, full, feats)
    k = a.n_total - len(binders)
    print(f"binders={len(binders)}  pool={len(nonb)}  selecting k={k}")
    if k > len(nonb):
        sys.exit(f"cannot select k={k} from a pool of {len(nonb)} "
                 f"(--max-hamming too tight, or --n-total too large)")
    if k < 1:
        sys.exit(f"--n-total {a.n_total} is <= the binder count {len(binders)}")
    print(f"mode={a.mode}: {len(names)} scores -> {len(cnames)} constraints\n")

    if a.report_bounds:
        lo, hi = bounds(C, k)
        bad = [(nm, l, h, t) for nm, l, h, t in zip(cnames, lo, hi, tgt)
               if not (l <= t <= h)]
        print(f"reachability at k={k}: {len(bad)}/{len(cnames)} constraints UNSATISFIABLE")
        for nm, l, h, t in bad:
            print(f"  {nm:30s} target {t:.3f} outside [{l:.3f}, {h:.3f}]")
        if not bad:
            print("  (all targets reachable -- the pool does not forbid the claim)")
        print()

    (mx, l2), sel, mean_c = optimize(C, tgt, k, a.restarts, a.passes, a.seed)
    allc = C.mean(0)
    # report per SCORE (the AUC column) rather than all constraints
    print(f"{'score':24s} {'vs ALL-NB':>10s} {'BALANCED':>10s}")
    for i, nm in enumerate(cnames):
        if nm.endswith("|AUC"):
            print(f"  {nm[:-4]:24s} {allc[i]:10.3f} {mean_c[i]:10.3f}")
    d = np.abs(mean_c - tgt)
    print(f"\nworst |achieved-target| over ALL {len(cnames)} constraints = {mx:.4f}"
          f"  (at {cnames[int(d.argmax())]};  sum sq = {l2:.5f})")

    ids = [nonb[j] for j in np.where(sel)[0]]
    if a.write:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["sequence_id", "class"])
            for s in binders:
                w.writerow([s, "binder"])
            for s in ids:
                w.writerow([s, "nonbinder"])
        print(f"wrote {a.out}  ({len(binders)} binders + {len(ids)} nonbinders)")
    else:
        print("(--write to emit the selection)")


if __name__ == "__main__":
    main()
