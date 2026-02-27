#!/usr/bin/env python3
"""
Cross-ligand scoring analysis for Boltz2 binary predictions.

Merges Boltz result CSVs with input label CSVs (pair_id + label),
evaluates per-metric discriminative power (AUC, Cohen's d), tests
combination strategies, and ranks metrics per-ligand and cross-ligand.

Usage:
    python scripts/analyze_boltz_scoring.py \
        --data LCA boltz_lca_results.csv boltz_lca_binary.csv \
        --data GLCA boltz_glca_results.csv boltz_glca_binary.csv \
        --data LCA-3-S boltz_lca3s_results.csv boltz_lca3s_binary.csv \
        --out-dir ml_modelling/analysis/boltz_LCA/scoring

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_results_csv(path):
    """Load Boltz results CSV into list of dicts with numeric conversion."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == 'name':
                    parsed[k] = v
                    continue
                try:
                    parsed[k] = float(v) if v else None
                except ValueError:
                    parsed[k] = None
            rows.append(parsed)
    return rows


def load_labels_csv(path):
    """Load labels CSV → {pair_id: bool}."""
    labels = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row.get('pair_id', row.get('name', '')).strip()
            label = float(row.get('label', 0))
            labels[pair_id] = label >= 0.5
    return labels


def merge_results_labels(results, labels):
    """Merge Boltz results with labels. Returns (merged_rows, n_binders, n_nonbinders)."""
    merged = []
    n_b, n_nb, n_missing = 0, 0, 0
    for row in results:
        name = row['name']
        if name not in labels:
            n_missing += 1
            continue
        row['is_binder'] = labels[name]
        if labels[name]:
            n_b += 1
        else:
            n_nb += 1
        merged.append(row)
    if n_missing > 0:
        print(f"  WARNING: {n_missing} predictions not found in labels CSV")
    return merged, n_b, n_nb


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def cohens_d(group1, group2):
    """Compute Cohen's d (positive = group1 > group2)."""
    if len(group1) < 2 or len(group2) < 2:
        return None
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = math.sqrt((s1**2 + s2**2) / 2)
    if pooled < 1e-12:
        return None
    return (m1 - m2) / pooled


def roc_auc(labels, scores):
    """Compute ROC AUC. Higher score = predicted binder.
    Returns (auc, optimal_threshold, sensitivity_at_opt, specificity_at_opt).
    """
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None, None, None
    pairs.sort(key=lambda x: -x[0])

    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None, None

    # Compute AUC + find optimal threshold (Youden's J)
    tp, fp = 0, 0
    auc = 0.0
    best_j = -1
    best_thr = None
    best_sens = 0
    best_spec = 0

    for i, (score, label) in enumerate(pairs):
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp

        sens = tp / n_pos
        spec = 1.0 - fp / n_neg
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_thr = score
            best_sens = sens
            best_spec = spec

    auc = auc / (n_pos * n_neg)
    return auc, best_thr, best_sens, best_spec


def bootstrap_auc(labels, scores, n_boot=1000, seed=42):
    """Bootstrap 95% CI for ROC AUC."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if len(pairs) < 10:
        return None, None
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(pairs), size=len(pairs))
        boot_scores = [pairs[i][0] for i in idx]
        boot_labels = [pairs[i][1] for i in idx]
        a, _, _, _ = roc_auc(boot_labels, boot_scores)
        if a is not None:
            aucs.append(a)
    if len(aucs) < 100:
        return None, None
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def enrichment_factor(labels, scores, top_fraction=0.1):
    """Enrichment factor at top X%: how many times more binders in top fraction vs random."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None
    pairs.sort(key=lambda x: -x[0])
    n_top = max(1, int(len(pairs) * top_fraction))
    top_binders = sum(1 for _, l in pairs[:n_top] if l)
    total_binders = sum(1 for _, l in pairs if l)
    if total_binders == 0:
        return None
    expected = total_binders * top_fraction
    if expected < 1e-12:
        return None
    return top_binders / expected


# ═══════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# (column_key, display_name, expected_sign)
# sign=+1: higher = more binder-like; sign=-1: lower = more binder-like
METRICS = [
    ('binary_iptm', 'ipTM', +1),
    ('binary_complex_plddt', 'Complex pLDDT', +1),
    ('binary_complex_iplddt', 'Interface pLDDT', +1),
    ('binary_plddt_protein', 'Protein pLDDT', +1),
    ('binary_plddt_pocket', 'Pocket pLDDT', +1),
    ('binary_plddt_ligand', 'Ligand pLDDT', +1),
    ('binary_complex_pde', 'Complex PDE', -1),
    ('binary_complex_ipde', 'Interface PDE', -1),
    ('binary_hbond_distance', 'H-bond distance', -1),
    ('binary_hbond_angle', 'H-bond angle', 0),  # not directional a priori
    ('binary_affinity_probability_binary', 'P(binder)', +1),
    ('binary_affinity_pred_value', 'Affinity pIC50', +1),
    ('binary_boltz_score', 'Boltz score', +1),
    ('binary_geometry_dist_score', 'Geom dist score', +1),
    ('binary_geometry_ang_score', 'Geom angle score', +1),
    ('binary_geometry_score', 'Geometry score', +1),
    ('binary_total_score', 'Total score', +1),
]


# ═══════════════════════════════════════════════════════════════════
# PER-LIGAND ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_single_ligand(ligand_name, rows, out_dir=None):
    """Full per-metric analysis for one ligand. Returns per-metric results dict."""
    binders = [r for r in rows if r['is_binder']]
    non_binders = [r for r in rows if not r['is_binder']]
    n_b, n_nb = len(binders), len(non_binders)

    print(f"\n{'='*70}")
    print(f"  {ligand_name}: {n_b} binders, {n_nb} non-binders ({n_b+n_nb} total)")
    print(f"{'='*70}")

    metric_results = {}

    for key, label, expected_sign in METRICS:
        b_vals = [r[key] for r in binders if r.get(key) is not None]
        nb_vals = [r[key] for r in non_binders if r.get(key) is not None]

        if not b_vals or not nb_vals:
            continue

        b_arr, nb_arr = np.array(b_vals), np.array(nb_vals)

        # Cohen's d (binder - non-binder)
        d = cohens_d(b_vals, nb_vals)

        # For AUC: orient scores so higher = more binder-like
        all_labels = [r['is_binder'] for r in rows if r.get(key) is not None]
        all_scores = [r[key] for r in rows if r.get(key) is not None]

        # Try both orientations, pick the one with AUC > 0.5
        auc_pos, thr_pos, sens_pos, spec_pos = roc_auc(all_labels, all_scores)
        auc_neg, thr_neg, sens_neg, spec_neg = roc_auc(
            all_labels, [-s for s in all_scores])

        if auc_pos is not None and auc_neg is not None:
            if auc_pos >= auc_neg:
                auc, thr, sens, spec = auc_pos, thr_pos, sens_pos, spec_pos
                best_sign = +1
            else:
                auc, thr, sens, spec = auc_neg, -thr_neg, sens_neg, spec_neg
                best_sign = -1
        elif auc_pos is not None:
            auc, thr, sens, spec = auc_pos, thr_pos, sens_pos, spec_pos
            best_sign = +1
        else:
            continue

        # Bootstrap CI
        oriented_scores = all_scores if best_sign == +1 else [-s for s in all_scores]
        ci_lo, ci_hi = bootstrap_auc(all_labels, oriented_scores, n_boot=500)

        # Enrichment factor at top 10%
        ef = enrichment_factor(all_labels, oriented_scores, top_fraction=0.1)

        # Direction check
        sign_match = "OK" if (expected_sign == 0 or best_sign == expected_sign) else "INVERTED"

        metric_results[key] = {
            'label': label,
            'auc': auc,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'd': d,
            'best_sign': best_sign,
            'expected_sign': expected_sign,
            'sign_match': sign_match,
            'optimal_thr': thr,
            'sensitivity': sens,
            'specificity': spec,
            'ef10': ef,
            'binder_mean': float(b_arr.mean()),
            'nonbinder_mean': float(nb_arr.mean()),
        }

    # Print ranking by AUC
    print(f"\n  {'Metric':<25} {'AUC':>6} {'95% CI':>15} {'d':>7} {'EF@10%':>7} {'Dir':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*15} {'-'*7} {'-'*7} {'-'*8}")

    ranked = sorted(metric_results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for key, m in ranked:
        ci_str = f"[{m['ci_lo']:.3f},{m['ci_hi']:.3f}]" if m['ci_lo'] is not None else "N/A"
        ef_str = f"{m['ef10']:.2f}" if m['ef10'] is not None else "N/A"
        d_str = f"{m['d']:+.3f}" if m['d'] is not None else "N/A"
        print(f"  {m['label']:<25} {m['auc']:.3f} {ci_str:>15} {d_str:>7} {ef_str:>7} {m['sign_match']:>8}")

    return metric_results


# ═══════════════════════════════════════════════════════════════════
# COMBINATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════

def compute_zscore_combination(rows, metric_keys, signs):
    """Compute z-score combination of specified metrics.
    Returns list of (score, is_binder) tuples."""
    # Compute mean/std for each metric
    stats = {}
    for key in metric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) < 2:
            continue
        stats[key] = (np.mean(vals), np.std(vals))

    scores = []
    for row in rows:
        z_total = 0.0
        n_valid = 0
        for key, sign in zip(metric_keys, signs):
            if key not in stats:
                continue
            v = row.get(key)
            if v is None:
                continue
            mu, sigma = stats[key]
            if sigma < 1e-12:
                continue
            z_total += sign * (v - mu) / sigma
            n_valid += 1
        if n_valid > 0:
            scores.append((z_total / n_valid, row['is_binder']))

    return scores


def evaluate_combination(rows, metric_keys, signs, name=""):
    """Evaluate a z-score combination and return AUC + stats."""
    scored = compute_zscore_combination(rows, metric_keys, signs)
    if len(scored) < 10:
        return None
    all_scores = [s for s, _ in scored]
    all_labels = [l for _, l in scored]
    auc, thr, sens, spec = roc_auc(all_labels, all_scores)
    if auc is None:
        return None
    d = cohens_d(
        [s for s, l in scored if l],
        [s for s, l in scored if not l]
    )
    ef = enrichment_factor(all_labels, all_scores, 0.1)
    return {
        'name': name,
        'auc': auc,
        'd': d,
        'ef10': ef,
        'sensitivity': sens,
        'specificity': spec,
        'n_metrics': len(metric_keys),
    }


def try_combinations(rows, metric_results, max_combo_size=5):
    """Try various metric combinations and return ranked results."""
    # Only use metrics with AUC > 0.5 (above random)
    useful = [(k, m) for k, m in metric_results.items() if m['auc'] > 0.52]
    useful.sort(key=lambda x: x[1]['auc'], reverse=True)

    if not useful:
        print("  No metrics with AUC > 0.52 — cannot build combinations")
        return []

    results = []

    # Strategy 1: Top-N by AUC
    for n in range(2, min(len(useful) + 1, max_combo_size + 1)):
        top_n = useful[:n]
        keys = [k for k, _ in top_n]
        signs = [m['best_sign'] for _, m in top_n]
        labels = [m['label'] for _, m in top_n]
        name = f"Top-{n}: {' + '.join(labels)}"
        r = evaluate_combination(rows, keys, signs, name)
        if r:
            results.append(r)

    # Strategy 2: Affinity-centric (P(binder) + pIC50 + geometry)
    affinity_keys = ['binary_affinity_probability_binary', 'binary_affinity_pred_value']
    geom_keys = ['binary_hbond_distance', 'binary_geometry_score']
    for combo_keys in [
        affinity_keys,
        affinity_keys + geom_keys,
        ['binary_affinity_probability_binary', 'binary_plddt_ligand'],
        ['binary_affinity_probability_binary', 'binary_plddt_ligand', 'binary_hbond_distance'],
    ]:
        valid_keys = [k for k in combo_keys if k in metric_results]
        if len(valid_keys) < 2:
            continue
        signs = [metric_results[k]['best_sign'] for k in valid_keys]
        labels = [metric_results[k]['label'] for k in valid_keys]
        name = f"Custom: {' + '.join(labels)}"
        r = evaluate_combination(rows, valid_keys, signs, name)
        if r:
            results.append(r)

    # Strategy 3: All-pairs of top-6 metrics
    top6 = useful[:6]
    for (k1, m1), (k2, m2) in combinations(top6, 2):
        name = f"Pair: {m1['label']} + {m2['label']}"
        r = evaluate_combination(
            rows,
            [k1, k2],
            [m1['best_sign'], m2['best_sign']],
            name
        )
        if r:
            results.append(r)

    # Deduplicate and sort
    seen = set()
    unique = []
    for r in results:
        if r['name'] not in seen:
            seen.add(r['name'])
            unique.append(r)
    unique.sort(key=lambda x: x['auc'], reverse=True)

    return unique


# ═══════════════════════════════════════════════════════════════════
# CROSS-LIGAND ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def cross_ligand_analysis(all_datasets):
    """Pool all ligands and evaluate universal scoring."""
    pooled = []
    for ligand, rows in all_datasets.items():
        for r in rows:
            r_copy = dict(r)
            r_copy['ligand'] = ligand
            pooled.append(r_copy)

    n_b = sum(1 for r in pooled if r['is_binder'])
    n_nb = len(pooled) - n_b

    print(f"\n{'='*70}")
    print(f"  CROSS-LIGAND POOLED: {n_b} binders, {n_nb} non-binders ({len(pooled)} total)")
    print(f"{'='*70}")

    metric_results = analyze_single_ligand("POOLED (all ligands)", pooled)

    print(f"\n  --- Combination strategies (pooled) ---")
    combos = try_combinations(pooled, metric_results)
    if combos:
        print(f"\n  {'Combination':<55} {'AUC':>6} {'d':>7} {'EF@10%':>7}")
        print(f"  {'-'*55} {'-'*6} {'-'*7} {'-'*7}")
        for c in combos[:15]:
            ef_str = f"{c['ef10']:.2f}" if c['ef10'] is not None else "N/A"
            d_str = f"{c['d']:+.3f}" if c['d'] is not None else "N/A"
            print(f"  {c['name']:<55} {c['auc']:.3f} {d_str:>7} {ef_str:>7}")

    return metric_results, combos


# ═══════════════════════════════════════════════════════════════════
# CONSISTENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def consistency_report(per_ligand_results):
    """Check which metrics are consistently good across all ligands."""
    ligands = list(per_ligand_results.keys())
    all_keys = set()
    for lr in per_ligand_results.values():
        all_keys.update(lr.keys())

    print(f"\n{'='*70}")
    print(f"  METRIC CONSISTENCY ACROSS LIGANDS")
    print(f"{'='*70}")

    consistent = []
    for key in all_keys:
        aucs = {}
        signs = {}
        for lig in ligands:
            if key in per_ligand_results[lig]:
                m = per_ligand_results[lig][key]
                aucs[lig] = m['auc']
                signs[lig] = m['best_sign']

        if len(aucs) < 2:
            continue

        mean_auc = np.mean(list(aucs.values()))
        min_auc = min(aucs.values())
        sign_consistent = len(set(signs.values())) == 1

        label = per_ligand_results[ligands[0]].get(key, {}).get('label', key)
        consistent.append({
            'key': key,
            'label': label,
            'mean_auc': mean_auc,
            'min_auc': min_auc,
            'sign_consistent': sign_consistent,
            'aucs': aucs,
            'signs': signs,
        })

    consistent.sort(key=lambda x: x['mean_auc'], reverse=True)

    auc_headers = ''.join(f'{lig:>8}' for lig in ligands)
    print(f"\n  {'Metric':<25} {'Mean':>6} {'Min':>6} {auc_headers} {'Sign':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8*len(ligands)} {'-'*6}")

    for c in consistent:
        auc_vals = ''.join(f"{c['aucs'].get(lig, 0):.3f}   " for lig in ligands)
        sign_str = "OK" if c['sign_consistent'] else "MIXED"
        print(f"  {c['label']:<25} {c['mean_auc']:.3f} {c['min_auc']:.3f}  {auc_vals} {sign_str:>6}")

    # Recommend best universal metrics
    good = [c for c in consistent if c['min_auc'] > 0.5 and c['sign_consistent']]
    if good:
        print(f"\n  RECOMMENDED UNIVERSAL METRICS (min AUC > 0.5, consistent sign):")
        for c in good[:5]:
            print(f"    {c['label']}: mean AUC={c['mean_auc']:.3f}, min AUC={c['min_auc']:.3f}")
    else:
        print(f"\n  WARNING: No metric has AUC > 0.5 across all ligands with consistent sign!")
        print(f"  Best available:")
        for c in consistent[:5]:
            print(f"    {c['label']}: mean AUC={c['mean_auc']:.3f}, min AUC={c['min_auc']:.3f}, sign={'OK' if c['sign_consistent'] else 'MIXED'}")

    return consistent


# ═══════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════

def write_metric_ranking_csv(per_ligand_results, pooled_results, out_path):
    """Write per-metric AUC/d/EF table as CSV."""
    fieldnames = ['metric', 'label']
    ligands = list(per_ligand_results.keys())
    for lig in ligands:
        fieldnames.extend([f'{lig}_auc', f'{lig}_d', f'{lig}_ef10', f'{lig}_sign'])
    fieldnames.extend(['pooled_auc', 'pooled_d', 'pooled_ef10', 'pooled_sign'])

    all_keys = set()
    for lr in per_ligand_results.values():
        all_keys.update(lr.keys())
    if pooled_results:
        all_keys.update(pooled_results.keys())

    rows = []
    for key in all_keys:
        row = {'metric': key}
        # Get label from any source
        for src in [pooled_results] + list(per_ligand_results.values()):
            if src and key in src:
                row['label'] = src[key]['label']
                break

        for lig in ligands:
            if key in per_ligand_results.get(lig, {}):
                m = per_ligand_results[lig][key]
                row[f'{lig}_auc'] = f"{m['auc']:.4f}" if m['auc'] is not None else ""
                row[f'{lig}_d'] = f"{m['d']:+.4f}" if m['d'] is not None else ""
                row[f'{lig}_ef10'] = f"{m['ef10']:.3f}" if m['ef10'] is not None else ""
                row[f'{lig}_sign'] = f"{m['best_sign']:+d}"
            else:
                row[f'{lig}_auc'] = ""
                row[f'{lig}_d'] = ""
                row[f'{lig}_ef10'] = ""
                row[f'{lig}_sign'] = ""

        if pooled_results and key in pooled_results:
            m = pooled_results[key]
            row['pooled_auc'] = f"{m['auc']:.4f}" if m['auc'] is not None else ""
            row['pooled_d'] = f"{m['d']:+.4f}" if m['d'] is not None else ""
            row['pooled_ef10'] = f"{m['ef10']:.3f}" if m['ef10'] is not None else ""
            row['pooled_sign'] = f"{m['best_sign']:+d}"

        rows.append(row)

    # Sort by pooled AUC descending
    def sort_key(r):
        v = r.get('pooled_auc', '')
        return float(v) if v else 0
    rows.sort(key=sort_key, reverse=True)

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f"\n  Wrote metric ranking to {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-ligand scoring analysis for Boltz2 binary predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/analyze_boltz_scoring.py \\
        --data LCA results_lca.csv labels_lca.csv \\
        --data GLCA results_glca.csv labels_glca.csv \\
        --data LCA-3-S results_lca3s.csv labels_lca3s.csv \\
        --out-dir analysis/scoring
        """)
    parser.add_argument("--data", nargs=3, action="append", required=True,
                        metavar=("LIGAND", "RESULTS_CSV", "LABELS_CSV"),
                        help="Ligand name, Boltz results CSV, and input labels CSV")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for CSV results (optional)")

    args = parser.parse_args()

    # Load all datasets
    all_datasets = {}
    for ligand, results_path, labels_path in args.data:
        print(f"\n--- Loading {ligand} ---")
        print(f"  Results: {results_path}")
        print(f"  Labels:  {labels_path}")

        results = load_results_csv(results_path)
        labels = load_labels_csv(labels_path)
        merged, n_b, n_nb = merge_results_labels(results, labels)

        print(f"  Merged: {len(merged)} rows ({n_b} binders, {n_nb} non-binders)")
        all_datasets[ligand] = merged

    # Per-ligand analysis
    per_ligand_results = {}
    per_ligand_combos = {}
    for ligand, rows in all_datasets.items():
        metric_results = analyze_single_ligand(ligand, rows)
        per_ligand_results[ligand] = metric_results

        print(f"\n  --- Combination strategies ({ligand}) ---")
        combos = try_combinations(rows, metric_results)
        per_ligand_combos[ligand] = combos
        if combos:
            print(f"\n  {'Combination':<55} {'AUC':>6} {'d':>7} {'EF@10%':>7}")
            print(f"  {'-'*55} {'-'*6} {'-'*7} {'-'*7}")
            for c in combos[:10]:
                ef_str = f"{c['ef10']:.2f}" if c['ef10'] is not None else "N/A"
                d_str = f"{c['d']:+.3f}" if c['d'] is not None else "N/A"
                print(f"  {c['name']:<55} {c['auc']:.3f} {d_str:>7} {ef_str:>7}")

    # Cross-ligand pooled analysis
    if len(all_datasets) > 1:
        pooled_results, pooled_combos = cross_ligand_analysis(all_datasets)
        consistency_report(per_ligand_results)
    else:
        pooled_results = None

    # Write output CSV if requested
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_metric_ranking_csv(per_ligand_results, pooled_results,
                                 out_dir / "metric_ranking.csv")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for ligand, results in per_ligand_results.items():
        ranked = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
        if ranked:
            best_key, best = ranked[0]
            print(f"\n  {ligand}:")
            print(f"    Best single metric: {best['label']} (AUC={best['auc']:.3f})")
            if per_ligand_combos.get(ligand):
                bc = per_ligand_combos[ligand][0]
                print(f"    Best combination:   {bc['name']} (AUC={bc['auc']:.3f})")

    if pooled_results:
        ranked = sorted(pooled_results.items(), key=lambda x: x[1]['auc'], reverse=True)
        if ranked:
            best_key, best = ranked[0]
            print(f"\n  POOLED:")
            print(f"    Best single metric: {best['label']} (AUC={best['auc']:.3f})")


if __name__ == "__main__":
    main()
