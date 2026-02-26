#!/usr/bin/env python3
"""Find the best combination of Boltz metrics to separate binders from non-binders.

Exhaustively searches all subsets of metrics using z-score combination,
with both equal weighting and Cohen's d weighting. Reports top combinations
by ROC AUC.

Usage:
    python scripts/find_best_boltz_combination.py results_binary_affinity.csv
"""

import sys
import csv
import numpy as np
from itertools import combinations
from pathlib import Path


def load_results(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k == 'name':
                    continue
                try:
                    row[k] = float(v) if v else None
                except ValueError:
                    row[k] = None
            rows.append(row)
    return rows


def classify_binder(name):
    num = int(name.split('_')[1])
    return num >= 3059


def compute_roc_auc(labels, scores):
    """ROC AUC without sklearn. Higher score = predicted binder."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tp, fp = 0, 0
    auc = 0.0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def compute_cohens_d(binder_vals, nonbinder_vals):
    if not binder_vals or not nonbinder_vals:
        return 0.0
    b = np.array(binder_vals)
    nb = np.array(nonbinder_vals)
    pooled_std = np.sqrt((b.std()**2 + nb.std()**2) / 2)
    if pooled_std < 1e-9:
        return 0.0
    return (b.mean() - nb.mean()) / pooled_std


def compute_combined_score(rows, metric_keys, signs, weights=None):
    """Compute z-score combination for given metrics.

    Returns list of (score, is_binder) for all rows with complete data.
    """
    n_metrics = len(metric_keys)
    if weights is None:
        weights = [1.0] * n_metrics

    # Precompute means and stds
    stats = []
    for key in metric_keys:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if len(vals) < 2:
            return []
        mu = np.mean(vals)
        sigma = np.std(vals)
        if sigma < 1e-9:
            return []
        stats.append((mu, sigma))

    results = []
    for row in rows:
        score = 0.0
        complete = True
        for i, key in enumerate(metric_keys):
            v = row.get(key)
            if v is None:
                complete = False
                break
            mu, sigma = stats[i]
            z = signs[i] * (v - mu) / sigma
            score += weights[i] * z
        if complete:
            score /= sum(weights)
            results.append((score, classify_binder(row['name'])))

    return results


def evaluate_combination(rows, metric_keys, signs, weights=None):
    """Return AUC and Cohen's d for a metric combination."""
    scored = compute_combined_score(rows, metric_keys, signs, weights)
    if len(scored) < 10:
        return None, None

    labels = [s[1] for s in scored]
    scores = [s[0] for s in scored]

    auc = compute_roc_auc(labels, scores)

    b_scores = [s for s, l in scored if l]
    nb_scores = [s for s, l in scored if not l]
    d = compute_cohens_d(b_scores, nb_scores)

    return auc, d


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_best_boltz_combination.py <results.csv>")
        sys.exit(1)

    rows = load_results(sys.argv[1])
    print(f"Loaded {len(rows)} predictions\n")

    n_binders = sum(1 for r in rows if classify_binder(r['name']))
    n_nonbinders = len(rows) - n_binders
    print(f"Binders: {n_binders}, Non-binders: {n_nonbinders}\n")

    # ── Define candidate metrics ──
    # (key, short_label, sign) — sign: +1 = higher is binder-like, -1 = lower is binder-like
    candidates = [
        ('binary_iptm', 'ipTM', +1),
        ('binary_complex_plddt', 'cpLDDT', +1),
        ('binary_complex_iplddt', 'ipLDDT', +1),
        ('binary_plddt_protein', 'prot_pLDDT', +1),
        ('binary_plddt_ligand', 'lig_pLDDT', +1),
        ('binary_complex_pde', 'cPDE', -1),
        ('binary_complex_ipde', 'iPDE', -1),
        ('binary_hbond_distance', 'hb_dist', -1),
        ('binary_hbond_angle', 'hb_ang', -1),  # try both signs
        ('binary_affinity_probability_binary', 'P(bind)', +1),
        ('binary_affinity_pred_value', 'pIC50', -1),  # empirically inverted
        ('binary_geometry_dist_score', 'geom_dist', +1),
        ('binary_geometry_ang_score', 'geom_ang', +1),
    ]

    # ── Single metric baselines ──
    print("=" * 70)
    print("SINGLE METRIC BASELINES:")
    print(f"  {'Metric':<20s} {'AUC':>6s} {'d':>7s}")
    print("-" * 40)

    single_results = []
    cohens_ds = {}
    for key, label, sign in candidates:
        labels = [classify_binder(r['name']) for r in rows]
        scores = [r.get(key) for r in rows]
        if sign == -1:
            adj_scores = [-s if s is not None else None for s in scores]
        else:
            adj_scores = scores
        auc = compute_roc_auc(labels, adj_scores)

        b_vals = [r.get(key) for r in rows if r.get(key) is not None and classify_binder(r['name'])]
        nb_vals = [r.get(key) for r in rows if r.get(key) is not None and not classify_binder(r['name'])]
        d = compute_cohens_d(b_vals, nb_vals) * sign  # sign-adjusted so positive = binder-discriminating
        cohens_ds[key] = abs(d)

        if auc is not None:
            single_results.append((auc, d, key, label))
            print(f"  {label:<20s} {auc:>6.3f} {d:>+7.3f}")

    single_results.sort(key=lambda x: -x[0])
    print()

    # ── Exhaustive subset search ──
    n = len(candidates)
    print(f"Searching all {2**n - 1} non-empty subsets of {n} metrics...")
    print()

    all_results = []

    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            keys = [candidates[i][0] for i in combo]
            labels_short = [candidates[i][1] for i in combo]
            signs = [candidates[i][2] for i in combo]

            # Equal weight
            auc_eq, d_eq = evaluate_combination(rows, keys, signs)

            # Cohen's d weighted (give more weight to individually discriminative metrics)
            d_weights = [max(cohens_ds.get(k, 0.1), 0.05) for k in keys]
            auc_dw, d_dw = evaluate_combination(rows, keys, signs, d_weights)

            if auc_eq is not None:
                all_results.append({
                    'keys': keys,
                    'labels': labels_short,
                    'signs': signs,
                    'size': size,
                    'auc_equal': auc_eq,
                    'd_equal': d_eq,
                    'auc_dweight': auc_dw,
                    'd_dweight': d_dw,
                    'best_auc': max(auc_eq, auc_dw or 0),
                    'best_weighting': 'equal' if auc_eq >= (auc_dw or 0) else 'd-weight',
                })

    all_results.sort(key=lambda x: -x['best_auc'])

    # ── Report top combinations by size ──
    for max_size in [1, 2, 3, 4, 5]:
        filtered = [r for r in all_results if r['size'] <= max_size]
        if not filtered:
            continue
        print("=" * 70)
        print(f"TOP 10 COMBINATIONS (up to {max_size} metrics):")
        print(f"  {'Rank':>4s}  {'AUC':>6s} {'d':>7s} {'Wt':>8s}  {'Metrics'}")
        print("-" * 70)
        for i, r in enumerate(filtered[:10]):
            metric_str = ' + '.join(r['labels'])
            print(f"  {i+1:>4d}  {r['best_auc']:>6.3f} "
                  f"{r['d_equal'] if r['best_weighting'] == 'equal' else r['d_dweight']:>+7.3f} "
                  f"{r['best_weighting']:>8s}  {metric_str}")
        print()

    # ── Overall top 25 ──
    print("=" * 70)
    print("OVERALL TOP 25 COMBINATIONS (any size):")
    print(f"  {'Rank':>4s}  {'AUC':>6s} {'d':>7s} {'N':>2s} {'Wt':>8s}  {'Metrics'}")
    print("-" * 70)
    for i, r in enumerate(all_results[:25]):
        metric_str = ' + '.join(r['labels'])
        d_val = r['d_equal'] if r['best_weighting'] == 'equal' else r['d_dweight']
        print(f"  {i+1:>4d}  {r['best_auc']:>6.3f} {d_val:>+7.3f} {r['size']:>2d} "
              f"{r['best_weighting']:>8s}  {metric_str}")
    print()

    # ── Best per size ──
    print("=" * 70)
    print("BEST COMBINATION PER SIZE:")
    print(f"  {'N':>2s}  {'AUC':>6s} {'d':>7s} {'Wt':>8s}  {'Metrics'}")
    print("-" * 70)
    seen_sizes = set()
    for r in all_results:
        if r['size'] not in seen_sizes:
            seen_sizes.add(r['size'])
            metric_str = ' + '.join(r['labels'])
            d_val = r['d_equal'] if r['best_weighting'] == 'equal' else r['d_dweight']
            print(f"  {r['size']:>2d}  {r['best_auc']:>6.3f} {d_val:>+7.3f} "
                  f"{r['best_weighting']:>8s}  {metric_str}")
    print()

    # ── Classification performance of the best combination ──
    best = all_results[0]
    print("=" * 70)
    print(f"BEST OVERALL: {' + '.join(best['labels'])} "
          f"(AUC={best['best_auc']:.3f}, weighting={best['best_weighting']})")
    print()

    keys = best['keys']
    signs = best['signs']
    if best['best_weighting'] == 'd-weight':
        weights = [max(cohens_ds.get(k, 0.1), 0.05) for k in keys]
    else:
        weights = None

    scored = compute_combined_score(rows, keys, signs, weights)

    b_scores = sorted([s for s, l in scored if l])
    nb_scores = sorted([s for s, l in scored if not l])

    print(f"  Binders:     mean={np.mean(b_scores):.3f}, median={np.median(b_scores):.3f}, "
          f"std={np.std(b_scores):.3f}")
    print(f"  Non-binders: mean={np.mean(nb_scores):.3f}, median={np.median(nb_scores):.3f}, "
          f"std={np.std(nb_scores):.3f}")
    print()

    # Build lookup for classification
    score_lookup = {}
    for row in rows:
        sc_list = compute_combined_score([row], keys, signs, weights)
        if sc_list:
            score_lookup[row['name']] = sc_list[0][0]

    print("  CLASSIFICATION (binder if score >= threshold):")
    all_scores = [s for s, _ in scored]
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        thr = np.percentile(all_scores, pct)
        tp = sum(1 for s, l in scored if l and s >= thr)
        fp = sum(1 for s, l in scored if not l and s >= thr)
        fn = sum(1 for s, l in scored if l and s < thr)
        tn = sum(1 for s, l in scored if not l and s < thr)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"    P{pct:02d} >= {thr:+.3f}: acc={acc:.2f}, prec={prec:.2f}, "
              f"recall={rec:.2f}, F1={f1:.2f}  (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
    print()

    # Top/bottom by best score
    print(f"  TOP 15 by best combination score:")
    scored_rows = [(score_lookup.get(r['name']), r) for r in rows if r['name'] in score_lookup]
    scored_rows.sort(key=lambda x: -x[0])
    for score, r in scored_rows[:15]:
        binder = "BINDER" if classify_binder(r['name']) else "non-binder"
        print(f"    {r['name']:12s}  score={score:+.3f}  {binder}")

    print()
    print(f"  BOTTOM 15 by best combination score:")
    for score, r in scored_rows[-15:]:
        binder = "BINDER" if classify_binder(r['name']) else "non-binder"
        print(f"    {r['name']:12s}  score={score:+.3f}  {binder}")


if __name__ == "__main__":
    main()
