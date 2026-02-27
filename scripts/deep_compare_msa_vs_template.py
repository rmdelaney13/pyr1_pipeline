#!/usr/bin/env python3
"""
Deep comparison of MSA-based vs template-based Boltz2 binary predictions.

Goes beyond simple AUC comparison to analyze:
  1. Bootstrap AUC with 95% CI for both approaches
  2. Per-variant paired analysis (rank agreement, sign-rank tests)
  3. Confusion matrix overlap at optimal thresholds
  4. Metric correlation structure within each approach
  5. Ensemble analysis (combining MSA + template metrics)
  6. Distribution shift analysis (how variance/separation change)
  7. Variant-level delta analysis (per-variant score changes)

Usage:
    python scripts/deep_compare_msa_vs_template.py \
        --msa-results results_all_affinity.csv \
        --template-results boltz_lca_binary_template_results.csv \
        --labels boltz_lca_binary.csv \
        --ligand LCA \
        --out-dir ml_modelling/analysis/boltz_LCA/msa_vs_template

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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════

if HAS_MPL:
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

MSA_COLOR = '#2196F3'
TMPL_COLOR = '#FF9800'
BINDER_COLOR = '#E53935'
NONBINDER_COLOR = '#90A4AE'
ENSEMBLE_COLOR = '#9C27B0'


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_results_csv(path):
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
    labels = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row.get('pair_id', row.get('name', '')).strip()
            label = float(row.get('label', 0))
            labels[pair_id] = label >= 0.5
    return labels


def merge(results, labels):
    merged = {}
    for row in results:
        name = row['name']
        if name in labels:
            row['is_binder'] = labels[name]
            merged[name] = row
    return merged


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def roc_auc(labels, scores):
    """ROC AUC. Returns (auc, threshold, sensitivity, specificity)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None, None, None
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None, None

    tp, fp, auc = 0, 0, 0.0
    best_j, best_thr, best_sens, best_spec = -1, None, 0, 0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
        sens = tp / n_pos
        spec = 1 - fp / n_neg
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = score
            best_sens = sens
            best_spec = spec

    auc /= (n_pos * n_neg)
    return auc, best_thr, best_sens, best_spec


def roc_auc_oriented(labels, scores):
    """ROC AUC auto-oriented so AUC >= 0.5. Returns (auc, flipped, threshold)."""
    auc, thr, sens, spec = roc_auc(labels, scores)
    if auc is None:
        return None, False, None
    if auc < 0.5:
        return 1 - auc, True, thr
    return auc, False, thr


def compute_roc_curve(labels, scores):
    """Full ROC curve. Returns (fprs, tprs, auc, flipped)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None, None, None, False
    pairs.sort(key=lambda x: -x[0])
    n_pos = sum(1 for _, l in pairs if l)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None, None, False

    fprs, tprs = [0.0], [0.0]
    tp, fp, auc = 0, 0, 0.0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
    auc /= (n_pos * n_neg)

    flipped = False
    if auc < 0.5:
        fprs = [1 - f for f in fprs]
        tprs = [1 - t for t in tprs]
        auc = 1 - auc
        flipped = True
    return np.array(fprs), np.array(tprs), auc, flipped


def bootstrap_auc(labels, scores, n_boot=2000, seed=42):
    """Bootstrap 95% CI for ROC AUC (oriented)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if len(pairs) < 10:
        return None, None, None
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(pairs), size=len(pairs))
        boot_s = [pairs[i][0] for i in idx]
        boot_l = [pairs[i][1] for i in idx]
        a, _, _, _ = roc_auc(boot_l, boot_s)
        if a is not None:
            aucs.append(max(a, 1 - a))  # oriented
    if len(aucs) < 100:
        return None, None, None
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def cohens_d(g1, g2):
    if len(g1) < 2 or len(g2) < 2:
        return None
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = math.sqrt((s1**2 + s2**2) / 2)
    if pooled < 1e-12:
        return None
    return (np.mean(g1) - np.mean(g2)) / pooled


def spearman_rank(x, y):
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return np.corrcoef(rx, ry)[0, 1]


def enrichment_factor(labels, scores, frac=0.1):
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None
    pairs.sort(key=lambda x: -x[0])
    n_top = max(1, int(len(pairs) * frac))
    top_b = sum(1 for _, l in pairs[:n_top] if l)
    total_b = sum(1 for _, l in pairs if l)
    if total_b == 0:
        return None
    expected = total_b * frac
    return top_b / expected if expected > 0 else None


def compute_zscore_combo(rows_list, metric_keys, signs):
    """Z-score combination. rows_list is list of dicts."""
    stats = {}
    for key in metric_keys:
        vals = [r[key] for r in rows_list if r.get(key) is not None]
        if len(vals) >= 2:
            stats[key] = (np.mean(vals), np.std(vals))
    scored = []
    for row in rows_list:
        z = 0.0
        n = 0
        for key, sign in zip(metric_keys, signs):
            if key not in stats:
                continue
            v = row.get(key)
            if v is None:
                continue
            mu, sigma = stats[key]
            if sigma < 1e-12:
                continue
            z += sign * (v - mu) / sigma
            n += 1
        if n > 0:
            scored.append((z / n, row['is_binder'], row.get('name', '')))
    return scored


# ═══════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

METRICS = [
    ('binary_iptm', 'ipTM'),
    ('binary_complex_plddt', 'Complex pLDDT'),
    ('binary_complex_iplddt', 'Interface pLDDT'),
    ('binary_plddt_protein', 'Protein pLDDT'),
    ('binary_plddt_pocket', 'Pocket pLDDT'),
    ('binary_plddt_ligand', 'Ligand pLDDT'),
    ('binary_complex_pde', 'Complex PDE'),
    ('binary_complex_ipde', 'Interface PDE'),
    ('binary_hbond_distance', 'H-bond dist'),
    ('binary_hbond_angle', 'H-bond angle'),
    ('binary_affinity_probability_binary', 'P(binder)'),
    ('binary_affinity_pred_value', 'Affinity pIC50'),
    ('binary_boltz_score', 'Boltz score'),
    ('binary_geometry_dist_score', 'Geom dist'),
    ('binary_geometry_ang_score', 'Geom angle'),
    ('binary_geometry_score', 'Geometry'),
    ('binary_total_score', 'Total score'),
]


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 1: BOOTSTRAP AUC COMPARISON
# ═══════════════════════════════════════════════════════════════════

def analysis_bootstrap_auc(msa_data, tmpl_data, shared_names):
    """Bootstrap AUC with 95% CI for both approaches, side-by-side."""
    print(f"\n{'='*80}")
    print(f"  SECTION 1: Bootstrap AUC Comparison (n=2000 resamples)")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<20} {'MSA AUC':>8} {'MSA 95%CI':>16} {'Tmpl AUC':>9} {'Tmpl 95%CI':>16} {'Sig?':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*16} {'-'*9} {'-'*16} {'-'*5}")

    results = []
    for key, label in METRICS:
        # MSA
        msa_labels = [msa_data[n]['is_binder'] for n in shared_names if msa_data[n].get(key) is not None]
        msa_scores = [msa_data[n][key] for n in shared_names if msa_data[n].get(key) is not None]
        msa_mean, msa_lo, msa_hi = bootstrap_auc(msa_labels, msa_scores)

        # Template
        tmpl_labels = [tmpl_data[n]['is_binder'] for n in shared_names if tmpl_data[n].get(key) is not None]
        tmpl_scores = [tmpl_data[n][key] for n in shared_names if tmpl_data[n].get(key) is not None]
        tmpl_mean, tmpl_lo, tmpl_hi = bootstrap_auc(tmpl_labels, tmpl_scores)

        if msa_mean is None or tmpl_mean is None:
            continue

        # Non-overlapping CIs = significant difference
        sig = "***" if msa_lo > tmpl_hi or tmpl_lo > msa_hi else ""

        msa_ci = f"[{msa_lo:.3f}, {msa_hi:.3f}]"
        tmpl_ci = f"[{tmpl_lo:.3f}, {tmpl_hi:.3f}]"
        print(f"  {label:<20} {msa_mean:>8.3f} {msa_ci:>16} {tmpl_mean:>9.3f} {tmpl_ci:>16} {sig:>5}")

        results.append({
            'key': key, 'label': label,
            'msa_auc': msa_mean, 'msa_lo': msa_lo, 'msa_hi': msa_hi,
            'tmpl_auc': tmpl_mean, 'tmpl_lo': tmpl_lo, 'tmpl_hi': tmpl_hi,
            'significant': bool(sig),
        })

    n_sig = sum(1 for r in results if r['significant'])
    print(f"\n  {n_sig}/{len(results)} metrics have non-overlapping 95% CIs")
    return results


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 2: RANK AGREEMENT
# ═══════════════════════════════════════════════════════════════════

def analysis_rank_agreement(msa_data, tmpl_data, shared_names):
    """How similarly do MSA and template rank variants?"""
    print(f"\n{'='*80}")
    print(f"  SECTION 2: Rank Agreement (Spearman correlation of variant rankings)")
    print(f"{'='*80}")

    print(f"\n  {'Metric':<20} {'Spearman r':>10} {'Pearson r':>10} {'Interpretation':>20}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*20}")

    results = []
    for key, label in METRICS:
        msa_vals, tmpl_vals = [], []
        for name in shared_names:
            mv = msa_data[name].get(key)
            tv = tmpl_data[name].get(key)
            if mv is not None and tv is not None:
                msa_vals.append(mv)
                tmpl_vals.append(tv)

        if len(msa_vals) < 10:
            continue

        sp = spearman_rank(msa_vals, tmpl_vals)
        pe = np.corrcoef(msa_vals, tmpl_vals)[0, 1]

        if sp is None:
            continue

        if abs(sp) > 0.7:
            interp = "Strong agreement"
        elif abs(sp) > 0.4:
            interp = "Moderate agreement"
        elif abs(sp) > 0.2:
            interp = "Weak agreement"
        else:
            interp = "NO agreement"

        print(f"  {label:<20} {sp:>10.3f} {pe:>10.3f} {interp:>20}")
        results.append({'key': key, 'label': label, 'spearman': sp, 'pearson': pe})

    return results


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 3: CONFUSION MATRIX OVERLAP
# ═══════════════════════════════════════════════════════════════════

def analysis_confusion_overlap(msa_data, tmpl_data, shared_names):
    """At optimal thresholds, what does each approach catch that the other misses?"""
    print(f"\n{'='*80}")
    print(f"  SECTION 3: Confusion Matrix Overlap at Optimal Thresholds")
    print(f"{'='*80}")

    key_metrics = [
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_iptm', 'ipTM'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_total_score', 'Total score'),
    ]

    results = []
    for key, label in key_metrics:
        # Get oriented AUC + threshold for each approach
        shared_list = list(shared_names)
        for source_name, source_data in [('MSA', msa_data), ('TMPL', tmpl_data)]:
            pass

        msa_labels = [msa_data[n]['is_binder'] for n in shared_list if msa_data[n].get(key) is not None]
        msa_scores = [msa_data[n][key] for n in shared_list if msa_data[n].get(key) is not None]
        msa_names_valid = [n for n in shared_list if msa_data[n].get(key) is not None]

        tmpl_labels = [tmpl_data[n]['is_binder'] for n in shared_list if tmpl_data[n].get(key) is not None]
        tmpl_scores = [tmpl_data[n][key] for n in shared_list if tmpl_data[n].get(key) is not None]
        tmpl_names_valid = [n for n in shared_list if tmpl_data[n].get(key) is not None]

        # Try both orientations for each
        msa_auc_pos, msa_thr_pos, _, _ = roc_auc(msa_labels, msa_scores)
        msa_auc_neg, msa_thr_neg, _, _ = roc_auc(msa_labels, [-s for s in msa_scores])

        tmpl_auc_pos, tmpl_thr_pos, _, _ = roc_auc(tmpl_labels, tmpl_scores)
        tmpl_auc_neg, tmpl_thr_neg, _, _ = roc_auc(tmpl_labels, [-s for s in tmpl_scores])

        if msa_auc_pos is None or tmpl_auc_pos is None:
            continue

        # Pick best orientation
        if msa_auc_pos >= (msa_auc_neg or 0):
            msa_thr = msa_thr_pos
            msa_sign = +1
        else:
            msa_thr = -msa_thr_neg
            msa_sign = -1

        if tmpl_auc_pos >= (tmpl_auc_neg or 0):
            tmpl_thr = tmpl_thr_pos
            tmpl_sign = +1
        else:
            tmpl_thr = -tmpl_thr_neg
            tmpl_sign = -1

        # Classify each shared variant
        # For names in both valid sets
        common = set(msa_names_valid) & set(tmpl_names_valid)
        msa_pos_set = set()
        tmpl_pos_set = set()
        binder_set = set()

        for name in common:
            is_b = msa_data[name]['is_binder']
            if is_b:
                binder_set.add(name)

            msa_val = msa_data[name][key] * msa_sign
            tmpl_val = tmpl_data[name][key] * tmpl_sign

            if msa_val >= msa_thr * msa_sign:
                msa_pos_set.add(name)
            if tmpl_val >= tmpl_thr * tmpl_sign:
                tmpl_pos_set.add(name)

        # Overlap stats
        both_pos = msa_pos_set & tmpl_pos_set
        msa_only = msa_pos_set - tmpl_pos_set
        tmpl_only = tmpl_pos_set - msa_pos_set
        neither = common - msa_pos_set - tmpl_pos_set

        # Among binders:
        b_both = binder_set & both_pos
        b_msa_only = binder_set & msa_only
        b_tmpl_only = binder_set & tmpl_only
        b_neither = binder_set & neither

        # Among non-binders:
        nb_set = common - binder_set
        nb_both = nb_set & both_pos  # false positives from both
        nb_msa_only = nb_set & msa_only  # FP from MSA only
        nb_tmpl_only = nb_set & tmpl_only  # FP from template only
        nb_neither = nb_set & neither  # true negatives from both

        print(f"\n  --- {label} ---")
        print(f"  MSA threshold: {msa_thr:.4f} (sign={msa_sign:+d}), Tmpl threshold: {tmpl_thr:.4f} (sign={tmpl_sign:+d})")
        print(f"  {len(common)} shared variants ({len(binder_set)} binders, {len(nb_set)} non-binders)")
        print(f"")
        print(f"  BINDERS detected (out of {len(binder_set)}):")
        print(f"    Both catch:       {len(b_both):>4}  ({100*len(b_both)/max(len(binder_set),1):.1f}%)")
        print(f"    MSA only:         {len(b_msa_only):>4}  ({100*len(b_msa_only)/max(len(binder_set),1):.1f}%)")
        print(f"    Template only:    {len(b_tmpl_only):>4}  ({100*len(b_tmpl_only)/max(len(binder_set),1):.1f}%)")
        print(f"    Neither catches:  {len(b_neither):>4}  ({100*len(b_neither)/max(len(binder_set),1):.1f}%)")
        print(f"")
        print(f"  NON-BINDERS (false positives, out of {len(nb_set)}):")
        print(f"    Both FP:          {len(nb_both):>4}  ({100*len(nb_both)/max(len(nb_set),1):.1f}%)")
        print(f"    MSA FP only:      {len(nb_msa_only):>4}  ({100*len(nb_msa_only)/max(len(nb_set),1):.1f}%)")
        print(f"    Tmpl FP only:     {len(nb_tmpl_only):>4}  ({100*len(nb_tmpl_only)/max(len(nb_set),1):.1f}%)")
        print(f"    Both TN:          {len(nb_neither):>4}  ({100*len(nb_neither)/max(len(nb_set),1):.1f}%)")

        # Union sensitivity
        union_binders = b_both | b_msa_only | b_tmpl_only
        union_fps = nb_both | nb_msa_only | nb_tmpl_only
        print(f"")
        print(f"  UNION (either catches): {len(union_binders)}/{len(binder_set)} binders ({100*len(union_binders)/max(len(binder_set),1):.1f}%), "
              f"{len(union_fps)} FPs")

        results.append({
            'label': label,
            'n_binders': len(binder_set),
            'b_both': len(b_both), 'b_msa_only': len(b_msa_only),
            'b_tmpl_only': len(b_tmpl_only), 'b_neither': len(b_neither),
            'union_binders': len(union_binders),
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 4: WITHIN-APPROACH METRIC CORRELATIONS
# ═══════════════════════════════════════════════════════════════════

def analysis_metric_correlations(msa_data, tmpl_data, shared_names):
    """Compare metric correlation structure between MSA and template."""
    print(f"\n{'='*80}")
    print(f"  SECTION 4: Metric Correlation Structure (MSA vs Template)")
    print(f"{'='*80}")

    key_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Cpx pLDDT'),
        ('binary_plddt_protein', 'Prot pLDDT'),
        ('binary_plddt_pocket', 'Pkt pLDDT'),
        ('binary_plddt_ligand', 'Lig pLDDT'),
        ('binary_hbond_distance', 'H-bond d'),
        ('binary_affinity_probability_binary', 'P(bind)'),
        ('binary_affinity_pred_value', 'pIC50'),
        ('binary_geometry_score', 'Geometry'),
    ]

    names = list(shared_names)

    for source_name, source_data in [('MSA', msa_data), ('TEMPLATE', tmpl_data)]:
        print(f"\n  --- {source_name} inter-metric correlations ---")

        # Build matrix
        header = "  " + "".join(f"{l:>10}" for _, l in key_metrics)
        print(header)

        for ki, (keyi, labeli) in enumerate(key_metrics):
            vals_i = [source_data[n].get(keyi) for n in names]
            row_str = f"  {labeli:<10}"

            for kj, (keyj, labelj) in enumerate(key_metrics):
                vals_j = [source_data[n].get(keyj) for n in names]

                # Filter to pairs where both are valid
                valid = [(vi, vj) for vi, vj in zip(vals_i, vals_j) if vi is not None and vj is not None]
                if len(valid) < 10:
                    row_str += f"{'N/A':>10}"
                    continue

                arr_i = np.array([v[0] for v in valid])
                arr_j = np.array([v[1] for v in valid])
                r = np.corrcoef(arr_i, arr_j)[0, 1]
                row_str += f"{r:>10.3f}"

            print(row_str)

    # Highlight biggest correlation differences
    print(f"\n  --- Largest correlation differences (MSA vs Template) ---")
    print(f"  {'Metric pair':<30} {'MSA r':>7} {'Tmpl r':>7} {'Delta':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}")

    diffs = []
    for (k1, l1), (k2, l2) in combinations(key_metrics, 2):
        v1_m = [msa_data[n].get(k1) for n in names]
        v2_m = [msa_data[n].get(k2) for n in names]
        v1_t = [tmpl_data[n].get(k1) for n in names]
        v2_t = [tmpl_data[n].get(k2) for n in names]

        valid_m = [(a, b) for a, b in zip(v1_m, v2_m) if a is not None and b is not None]
        valid_t = [(a, b) for a, b in zip(v1_t, v2_t) if a is not None and b is not None]

        if len(valid_m) < 10 or len(valid_t) < 10:
            continue

        r_m = np.corrcoef([v[0] for v in valid_m], [v[1] for v in valid_m])[0, 1]
        r_t = np.corrcoef([v[0] for v in valid_t], [v[1] for v in valid_t])[0, 1]
        diffs.append((f"{l1} × {l2}", r_m, r_t, abs(r_t - r_m)))

    diffs.sort(key=lambda x: x[3], reverse=True)
    for pair, rm, rt, delta in diffs[:10]:
        print(f"  {pair:<30} {rm:>7.3f} {rt:>7.3f} {abs(rt-rm):>+7.3f}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 5: ENSEMBLE (MSA + TEMPLATE COMBINED)
# ═══════════════════════════════════════════════════════════════════

def analysis_ensemble(msa_data, tmpl_data, shared_names):
    """Test whether combining MSA + template metrics improves discrimination."""
    print(f"\n{'='*80}")
    print(f"  SECTION 5: Ensemble Analysis (MSA + Template Combined)")
    print(f"{'='*80}")

    names = list(shared_names)

    # Build unified rows with both MSA and template features
    unified = []
    for name in names:
        row = {'name': name, 'is_binder': msa_data[name]['is_binder']}
        for key, _ in METRICS:
            row[f'msa_{key}'] = msa_data[name].get(key)
            row[f'tmpl_{key}'] = tmpl_data[name].get(key)
        unified.append(row)

    # Single-source best combos
    print(f"\n  --- Reference: best single-source combos ---")

    # MSA-only z-score combos
    msa_rows = list(msa_data.values())
    tmpl_rows = list(tmpl_data.values())

    # Find best single metric from each
    best_msa, best_tmpl = None, None
    for key, label in METRICS:
        for source_name, source_rows in [('MSA', msa_rows), ('TMPL', tmpl_rows)]:
            labels_arr = [r['is_binder'] for r in source_rows if r.get(key) is not None]
            scores_arr = [r[key] for r in source_rows if r.get(key) is not None]
            auc, flipped, _ = roc_auc_oriented(labels_arr, scores_arr)
            if auc is None:
                continue
            sign = -1 if flipped else +1
            if source_name == 'MSA':
                if best_msa is None or auc > best_msa[1]:
                    best_msa = (key, auc, label, sign)
            else:
                if best_tmpl is None or auc > best_tmpl[1]:
                    best_tmpl = (key, auc, label, sign)

    if best_msa:
        print(f"  Best MSA single:  {best_msa[2]} AUC={best_msa[1]:.3f}")
    if best_tmpl:
        print(f"  Best Tmpl single: {best_tmpl[2]} AUC={best_tmpl[1]:.3f}")

    # Now test cross-source ensembles
    print(f"\n  --- Cross-source ensembles ---")
    print(f"  {'Ensemble':<55} {'AUC':>6} {'d':>7} {'EF@10%':>7}")
    print(f"  {'-'*55} {'-'*6} {'-'*7} {'-'*7}")

    # Define ensemble combos to test
    ensemble_combos = []

    # MSA P(binder) + Template geometry metrics
    ensemble_combos.append((
        "MSA P(binder) + Tmpl H-bond dist",
        [('msa_binary_affinity_probability_binary', +1),
         ('tmpl_binary_hbond_distance', -1)],
    ))
    ensemble_combos.append((
        "MSA P(binder) + Tmpl H-bond angle",
        [('msa_binary_affinity_probability_binary', +1),
         ('tmpl_binary_hbond_angle', -1)],
    ))
    ensemble_combos.append((
        "MSA P(binder) + Tmpl Geometry",
        [('msa_binary_affinity_probability_binary', +1),
         ('tmpl_binary_geometry_score', +1)],
    ))

    # MSA confidence + Template geometry
    ensemble_combos.append((
        "MSA Complex pLDDT + Tmpl H-bond dist",
        [('msa_binary_complex_plddt', +1),
         ('tmpl_binary_hbond_distance', -1)],
    ))
    ensemble_combos.append((
        "MSA Interface pLDDT + Tmpl H-bond dist",
        [('msa_binary_complex_iplddt', +1),
         ('tmpl_binary_hbond_distance', -1)],
    ))
    ensemble_combos.append((
        "MSA ipTM + Tmpl H-bond dist + Tmpl Geometry",
        [('msa_binary_iptm', +1),
         ('tmpl_binary_hbond_distance', -1),
         ('tmpl_binary_geometry_score', +1)],
    ))

    # Full kitchen sink
    ensemble_combos.append((
        "MSA P(binder) + MSA ipTM + Tmpl H-bond d + Tmpl Geom",
        [('msa_binary_affinity_probability_binary', +1),
         ('msa_binary_iptm', +1),
         ('tmpl_binary_hbond_distance', -1),
         ('tmpl_binary_geometry_score', +1)],
    ))
    ensemble_combos.append((
        "MSA Iface pLDDT + MSA P(bind) + Tmpl H-bond d + Tmpl ang",
        [('msa_binary_complex_iplddt', +1),
         ('msa_binary_affinity_probability_binary', +1),
         ('tmpl_binary_hbond_distance', -1),
         ('tmpl_binary_hbond_angle', -1)],
    ))

    # MSA-only combos for reference
    ensemble_combos.append((
        "[MSA only] Iface pLDDT + H-bond dist",
        [('msa_binary_complex_iplddt', +1),
         ('msa_binary_hbond_distance', -1)],
    ))
    ensemble_combos.append((
        "[MSA only] Top-4: ipTM + CpLDDT + IpLDDT + P(bind)",
        [('msa_binary_iptm', +1),
         ('msa_binary_complex_plddt', +1),
         ('msa_binary_complex_iplddt', +1),
         ('msa_binary_affinity_probability_binary', +1)],
    ))

    # Template-only combos for reference
    ensemble_combos.append((
        "[Tmpl only] H-bond dist + angle + Geom",
        [('tmpl_binary_hbond_distance', -1),
         ('tmpl_binary_hbond_angle', -1),
         ('tmpl_binary_geometry_score', +1)],
    ))
    ensemble_combos.append((
        "[Tmpl only] Top-5 from template scoring",
        [('tmpl_binary_hbond_distance', -1),
         ('tmpl_binary_hbond_angle', -1),
         ('tmpl_binary_geometry_score', +1),
         ('tmpl_binary_complex_pde', -1),
         ('tmpl_binary_complex_ipde', -1)],
    ))

    combo_results = []
    for combo_name, combo_spec in ensemble_combos:
        keys = [k for k, _ in combo_spec]
        signs = [s for _, s in combo_spec]
        scored = compute_zscore_combo(unified, keys, signs)
        if len(scored) < 10:
            continue

        all_scores = [s for s, _, _ in scored]
        all_labels = [l for _, l, _ in scored]
        auc, _, _, _ = roc_auc(all_labels, all_scores)
        if auc is None:
            continue
        auc = max(auc, 1 - auc)

        d = cohens_d(
            [s for s, l, _ in scored if l],
            [s for s, l, _ in scored if not l]
        )
        ef = enrichment_factor(all_labels, all_scores, 0.1)

        d_str = f"{d:+.3f}" if d is not None else "N/A"
        ef_str = f"{ef:.2f}" if ef is not None else "N/A"
        print(f"  {combo_name:<55} {auc:.3f} {d_str:>7} {ef_str:>7}")
        combo_results.append({'name': combo_name, 'auc': auc, 'd': d, 'ef10': ef})

    combo_results.sort(key=lambda x: x['auc'], reverse=True)

    if combo_results:
        best = combo_results[0]
        print(f"\n  BEST ENSEMBLE: {best['name']} — AUC={best['auc']:.3f}")

    return combo_results


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 6: DISTRIBUTION SHIFT (VARIANCE & SEPARATION)
# ═══════════════════════════════════════════════════════════════════

def analysis_distribution_shift(msa_data, tmpl_data, shared_names):
    """How do distributions change between approaches?"""
    print(f"\n{'='*80}")
    print(f"  SECTION 6: Distribution Shift Analysis")
    print(f"{'='*80}")

    print(f"\n  How mean, std, and binder-nonbinder separation change between MSA and template")
    print(f"")

    header = (f"  {'Metric':<18} "
              f"{'MSA B mean':>10} {'MSA NB mean':>11} {'MSA B std':>9} {'MSA NB std':>10} "
              f"{'Tmpl B mean':>11} {'Tmpl NB mean':>12} {'Tmpl B std':>10} {'Tmpl NB std':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    names = list(shared_names)
    binder_names = [n for n in names if msa_data[n]['is_binder']]
    nonbinder_names = [n for n in names if not msa_data[n]['is_binder']]

    key_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Cpx pLDDT'),
        ('binary_complex_iplddt', 'Iface pLDDT'),
        ('binary_plddt_protein', 'Prot pLDDT'),
        ('binary_plddt_pocket', 'Pkt pLDDT'),
        ('binary_plddt_ligand', 'Lig pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_hbond_angle', 'H-bond angle'),
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_affinity_pred_value', 'pIC50'),
        ('binary_geometry_score', 'Geometry'),
    ]

    for key, label in key_metrics:
        msa_b = [msa_data[n][key] for n in binder_names if msa_data[n].get(key) is not None]
        msa_nb = [msa_data[n][key] for n in nonbinder_names if msa_data[n].get(key) is not None]
        tmpl_b = [tmpl_data[n][key] for n in binder_names if tmpl_data[n].get(key) is not None]
        tmpl_nb = [tmpl_data[n][key] for n in nonbinder_names if tmpl_data[n].get(key) is not None]

        if not msa_b or not msa_nb or not tmpl_b or not tmpl_nb:
            continue

        print(f"  {label:<18} "
              f"{np.mean(msa_b):>10.4f} {np.mean(msa_nb):>11.4f} {np.std(msa_b):>9.4f} {np.std(msa_nb):>10.4f} "
              f"{np.mean(tmpl_b):>11.4f} {np.mean(tmpl_nb):>12.4f} {np.std(tmpl_b):>10.4f} {np.std(tmpl_nb):>11.4f}")

    # Variance ratio analysis
    print(f"\n  --- Variance changes (std ratio: template/MSA) ---")
    print(f"  {'Metric':<18} {'Binder std ratio':>16} {'NB std ratio':>12} {'Interpretation':<30}")
    print(f"  {'-'*18} {'-'*16} {'-'*12} {'-'*30}")

    for key, label in key_metrics:
        msa_b = [msa_data[n][key] for n in binder_names if msa_data[n].get(key) is not None]
        msa_nb = [msa_data[n][key] for n in nonbinder_names if msa_data[n].get(key) is not None]
        tmpl_b = [tmpl_data[n][key] for n in binder_names if tmpl_data[n].get(key) is not None]
        tmpl_nb = [tmpl_data[n][key] for n in nonbinder_names if tmpl_data[n].get(key) is not None]

        if not msa_b or not msa_nb or not tmpl_b or not tmpl_nb:
            continue

        msa_b_std = np.std(msa_b)
        msa_nb_std = np.std(msa_nb)
        tmpl_b_std = np.std(tmpl_b)
        tmpl_nb_std = np.std(tmpl_nb)

        b_ratio = tmpl_b_std / msa_b_std if msa_b_std > 1e-10 else float('inf')
        nb_ratio = tmpl_nb_std / msa_nb_std if msa_nb_std > 1e-10 else float('inf')

        if b_ratio < 0.5 and nb_ratio < 0.5:
            interp = "Tmpl compresses BOTH"
        elif b_ratio < 0.5:
            interp = "Tmpl compresses binders"
        elif nb_ratio < 0.5:
            interp = "Tmpl compresses non-binders"
        elif b_ratio > 2.0 or nb_ratio > 2.0:
            interp = "Tmpl EXPANDS variance"
        else:
            interp = "Similar variance"

        print(f"  {label:<18} {b_ratio:>16.3f} {nb_ratio:>12.3f} {interp:<30}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 7: VARIANT-LEVEL DELTAS
# ═══════════════════════════════════════════════════════════════════

def analysis_variant_deltas(msa_data, tmpl_data, shared_names):
    """Per-variant score changes between approaches. Identify outliers."""
    print(f"\n{'='*80}")
    print(f"  SECTION 7: Variant-Level Score Deltas (Template - MSA)")
    print(f"{'='*80}")

    names = list(shared_names)
    binder_names = [n for n in names if msa_data[n]['is_binder']]
    nonbinder_names = [n for n in names if not msa_data[n]['is_binder']]

    key_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Cpx pLDDT'),
        ('binary_complex_iplddt', 'Iface pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_geometry_score', 'Geometry'),
    ]

    print(f"\n  Mean per-variant delta (Template - MSA), split by binder status:")
    print(f"  {'Metric':<18} {'Binder delta':>12} {'NB delta':>12} {'Diff':>8} {'Helps binders?':>15}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*8} {'-'*15}")

    for key, label in key_metrics:
        b_deltas = []
        nb_deltas = []
        for name in binder_names:
            mv = msa_data[name].get(key)
            tv = tmpl_data[name].get(key)
            if mv is not None and tv is not None:
                b_deltas.append(tv - mv)
        for name in nonbinder_names:
            mv = msa_data[name].get(key)
            tv = tmpl_data[name].get(key)
            if mv is not None and tv is not None:
                nb_deltas.append(tv - mv)

        if not b_deltas or not nb_deltas:
            continue

        b_mean = np.mean(b_deltas)
        nb_mean = np.mean(nb_deltas)
        diff = b_mean - nb_mean

        # For "higher = better" metrics, positive diff means template helps binders more
        # For H-bond distance, lower = better, so negative diff means template helps binders more
        if 'distance' in label.lower():
            helps = "YES" if diff < 0 else "no"
        else:
            helps = "YES" if diff > 0 else "no"

        print(f"  {label:<18} {b_mean:>+12.4f} {nb_mean:>+12.4f} {diff:>+8.4f} {helps:>15}")

    # Show top outlier variants
    print(f"\n  --- Top 10 binders most IMPROVED by template (ipTM) ---")
    iptm_key = 'binary_iptm'
    binder_deltas = []
    for name in binder_names:
        mv = msa_data[name].get(iptm_key)
        tv = tmpl_data[name].get(iptm_key)
        if mv is not None and tv is not None:
            binder_deltas.append((name, tv - mv, mv, tv))

    binder_deltas.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Variant':<15} {'MSA ipTM':>10} {'Tmpl ipTM':>10} {'Delta':>8}")
    for name, delta, mv, tv in binder_deltas[:10]:
        print(f"  {name:<15} {mv:>10.4f} {tv:>10.4f} {delta:>+8.4f}")

    print(f"\n  --- Top 10 binders most HURT by template (ipTM) ---")
    for name, delta, mv, tv in binder_deltas[-10:]:
        print(f"  {name:<15} {mv:>10.4f} {tv:>10.4f} {delta:>+8.4f}")


# ═══════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════

def make_figures(msa_data, tmpl_data, shared_names, bootstrap_results, rank_results,
                 ensemble_results, ligand, out_dir):
    """Generate comprehensive comparison figures."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping figures")
        return

    names = list(shared_names)
    binder_names = [n for n in names if msa_data[n]['is_binder']]
    nonbinder_names = [n for n in names if not msa_data[n]['is_binder']]

    # ── Figure 1: Bootstrap AUC with CIs ──
    if bootstrap_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels_list = [r['label'] for r in bootstrap_results]
        msa_aucs = [r['msa_auc'] for r in bootstrap_results]
        tmpl_aucs = [r['tmpl_auc'] for r in bootstrap_results]
        msa_errs = [[r['msa_auc'] - r['msa_lo'], r['msa_hi'] - r['msa_auc']] for r in bootstrap_results]
        tmpl_errs = [[r['tmpl_auc'] - r['tmpl_lo'], r['tmpl_hi'] - r['tmpl_auc']] for r in bootstrap_results]

        # Sort by MSA AUC
        sort_idx = np.argsort([-a for a in msa_aucs])
        labels_list = [labels_list[i] for i in sort_idx]
        msa_aucs = [msa_aucs[i] for i in sort_idx]
        tmpl_aucs = [tmpl_aucs[i] for i in sort_idx]
        msa_errs = [msa_errs[i] for i in sort_idx]
        tmpl_errs = [tmpl_errs[i] for i in sort_idx]

        x = np.arange(len(labels_list))
        w = 0.35
        ax.bar(x - w/2, msa_aucs, w, label='MSA', color=MSA_COLOR, alpha=0.8,
               yerr=np.array(msa_errs).T, capsize=3, error_kw={'linewidth': 0.8})
        ax.bar(x + w/2, tmpl_aucs, w, label='Template', color=TMPL_COLOR, alpha=0.8,
               yerr=np.array(tmpl_errs).T, capsize=3, error_kw={'linewidth': 0.8})
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # Mark significant differences
        for i, r in enumerate(bootstrap_results):
            si = sort_idx[i] if i < len(sort_idx) else i
            if bootstrap_results[si]['significant']:
                ax.text(i, max(msa_aucs[i], tmpl_aucs[i]) + 0.02, '*',
                        ha='center', fontsize=14, color='red', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('ROC AUC (bootstrap mean)')
        ax.set_ylim(0.4, 0.85)
        ax.legend()
        ax.set_title(f'{ligand}: MSA vs Template — Bootstrap AUC with 95% CI')
        fig.tight_layout()
        fig.savefig(out_dir / 'fig1_bootstrap_auc.png')
        plt.close(fig)
        print(f"  Saved fig1_bootstrap_auc.png")

    # ── Figure 2: Multi-panel ROC for key metrics ──
    key_metrics = [
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_iptm', 'ipTM'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_complex_iplddt', 'Interface pLDDT'),
        ('binary_geometry_score', 'Geometry'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(key_metrics):
        ax = axes[idx]
        for source_name, source_data, color in [
            ('MSA', msa_data, MSA_COLOR),
            ('Template', tmpl_data, TMPL_COLOR),
        ]:
            src_labels = [source_data[n]['is_binder'] for n in names if source_data[n].get(key) is not None]
            src_scores = [source_data[n][key] for n in names if source_data[n].get(key) is not None]
            fprs, tprs, auc, flipped = compute_roc_curve(src_labels, src_scores)
            if fprs is not None:
                flip_note = " (inv)" if flipped else ""
                ax.plot(fprs, tprs, color=color, linewidth=1.8,
                        label=f'{source_name}: {auc:.3f}{flip_note}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(label)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    fig.suptitle(f'{ligand}: MSA vs Template ROC Comparison', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig2_roc_comparison.png')
    plt.close(fig)
    print(f"  Saved fig2_roc_comparison.png")

    # ── Figure 3: Paired scatter (MSA vs Template values) ──
    scatter_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_affinity_probability_binary', 'P(binder)'),
        ('binary_plddt_ligand', 'Ligand pLDDT'),
        ('binary_geometry_score', 'Geometry'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(scatter_metrics):
        ax = axes[idx]

        for name_list, color, alpha, size, zorder, cat_label in [
            (nonbinder_names, NONBINDER_COLOR, 0.3, 8, 1, 'Non-binder'),
            (binder_names, BINDER_COLOR, 0.8, 25, 2, 'Binder'),
        ]:
            xs, ys = [], []
            for name in name_list:
                mv = msa_data[name].get(key)
                tv = tmpl_data[name].get(key)
                if mv is not None and tv is not None:
                    xs.append(mv)
                    ys.append(tv)
            ax.scatter(xs, ys, s=size, alpha=alpha, color=color,
                       edgecolors='none', zorder=zorder, label=cat_label)

        # Diagonal
        lims = [ax.get_xlim(), ax.get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=0.8)
        ax.set_xlabel(f'MSA {label}')
        ax.set_ylabel(f'Template {label}')
        ax.set_title(label)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    fig.suptitle(f'{ligand}: Paired Values (MSA vs Template)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig3_paired_scatter.png')
    plt.close(fig)
    print(f"  Saved fig3_paired_scatter.png")

    # ── Figure 4: Distribution comparison (binder vs non-binder, MSA vs Template) ──
    dist_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_affinity_probability_binary', 'P(binder)'),
    ]

    fig, axes = plt.subplots(len(dist_metrics), 2, figsize=(10, 3 * len(dist_metrics)))

    for row, (key, label) in enumerate(dist_metrics):
        for col, (source_name, source_data) in enumerate([('MSA', msa_data), ('Template', tmpl_data)]):
            ax = axes[row, col]

            b_vals = [source_data[n][key] for n in binder_names if source_data[n].get(key) is not None]
            nb_vals = [source_data[n][key] for n in nonbinder_names if source_data[n].get(key) is not None]

            if not b_vals or not nb_vals:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            all_vals = b_vals + nb_vals
            bins = np.linspace(min(all_vals), max(all_vals), 40)
            ax.hist(nb_vals, bins=bins, alpha=0.6, color=NONBINDER_COLOR,
                    label=f'NB (n={len(nb_vals)})', density=True)
            ax.hist(b_vals, bins=bins, alpha=0.7, color=BINDER_COLOR,
                    label=f'B (n={len(b_vals)})', density=True)

            auc, flipped, _ = roc_auc_oriented(
                [True]*len(b_vals) + [False]*len(nb_vals),
                b_vals + nb_vals)
            flip_note = " inv" if flipped else ""
            ax.set_title(f'{source_name} {label} (AUC={auc:.3f}{flip_note})', fontsize=10)
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel('Density')

    fig.suptitle(f'{ligand}: Distribution Comparison', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig4_distributions.png')
    plt.close(fig)
    print(f"  Saved fig4_distributions.png")

    # ── Figure 5: Rank agreement scatter (Spearman) ──
    if rank_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels_list = [r['label'] for r in rank_results]
        spearmans = [r['spearman'] for r in rank_results]
        pearsons = [r['pearson'] for r in rank_results]

        sort_idx = np.argsort(spearmans)
        labels_list = [labels_list[i] for i in sort_idx]
        spearmans = [spearmans[i] for i in sort_idx]
        pearsons = [pearsons[i] for i in sort_idx]

        y = np.arange(len(labels_list))
        ax.barh(y, spearmans, height=0.6, color=MSA_COLOR, alpha=0.7, label='Spearman')
        ax.scatter(pearsons, y, color=TMPL_COLOR, s=40, zorder=3, label='Pearson')

        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_list, fontsize=9)
        ax.set_xlabel('Correlation (MSA vs Template variant rankings)')
        ax.set_title(f'{ligand}: How Similarly Do MSA and Template Rank Variants?')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.3, 1.0)
        fig.tight_layout()
        fig.savefig(out_dir / 'fig5_rank_agreement.png')
        plt.close(fig)
        print(f"  Saved fig5_rank_agreement.png")

    # ── Figure 6: Ensemble AUC comparison ──
    if ensemble_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_combos = sorted(ensemble_results, key=lambda x: x['auc'], reverse=True)[:12]

        names_list = [c['name'] for c in sorted_combos]
        aucs = [c['auc'] for c in sorted_combos]

        colors = []
        for name in names_list:
            if name.startswith('[MSA only]'):
                colors.append(MSA_COLOR)
            elif name.startswith('[Tmpl only]'):
                colors.append(TMPL_COLOR)
            else:
                colors.append(ENSEMBLE_COLOR)

        y = np.arange(len(names_list))
        ax.barh(y, aucs, color=colors, alpha=0.8)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        for i, auc in enumerate(aucs):
            ax.text(auc + 0.003, i, f'{auc:.3f}', va='center', fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(names_list, fontsize=8)
        ax.set_xlabel('ROC AUC')
        ax.set_xlim(0.45, max(aucs) + 0.06)
        ax.set_title(f'{ligand}: Ensemble Analysis — MSA + Template Combined')

        ax.legend(handles=[
            Patch(facecolor=MSA_COLOR, label='MSA only'),
            Patch(facecolor=TMPL_COLOR, label='Template only'),
            Patch(facecolor=ENSEMBLE_COLOR, label='Cross-source ensemble'),
        ], loc='lower right', fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / 'fig6_ensemble_comparison.png')
        plt.close(fig)
        print(f"  Saved fig6_ensemble_comparison.png")

    # ── Figure 7: Delta distributions (Template - MSA) for binders vs non-binders ──
    delta_metrics = [
        ('binary_iptm', 'ipTM'),
        ('binary_complex_plddt', 'Complex pLDDT'),
        ('binary_hbond_distance', 'H-bond dist'),
        ('binary_affinity_probability_binary', 'P(binder)'),
    ]

    fig, axes = plt.subplots(1, len(delta_metrics), figsize=(4 * len(delta_metrics), 4))

    for idx, (key, label) in enumerate(delta_metrics):
        ax = axes[idx]
        b_deltas = [tmpl_data[n][key] - msa_data[n][key]
                     for n in binder_names
                     if msa_data[n].get(key) is not None and tmpl_data[n].get(key) is not None]
        nb_deltas = [tmpl_data[n][key] - msa_data[n][key]
                      for n in nonbinder_names
                      if msa_data[n].get(key) is not None and tmpl_data[n].get(key) is not None]

        if not b_deltas or not nb_deltas:
            continue

        parts = ax.violinplot([nb_deltas, b_deltas], positions=[0, 1],
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(NONBINDER_COLOR if i == 0 else BINDER_COLOR)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-binder', 'Binder'])
        ax.set_ylabel(f'\u0394 {label} (Tmpl - MSA)')
        ax.set_title(label)

    fig.suptitle(f'{ligand}: Per-Variant Score Deltas (Template - MSA)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig7_delta_distributions.png')
    plt.close(fig)
    print(f"  Saved fig7_delta_distributions.png")


# ═══════════════════════════════════════════════════════════════════
# OUTPUT CSV
# ═══════════════════════════════════════════════════════════════════

def write_paired_csv(msa_data, tmpl_data, shared_names, out_path):
    """Write per-variant paired comparison CSV."""
    names = sorted(shared_names)

    fieldnames = ['name', 'is_binder']
    for key, label in METRICS:
        fieldnames.extend([f'msa_{key}', f'tmpl_{key}', f'delta_{key}'])

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name in names:
            row = {
                'name': name,
                'is_binder': 1 if msa_data[name]['is_binder'] else 0,
            }
            for key, _ in METRICS:
                mv = msa_data[name].get(key)
                tv = tmpl_data[name].get(key)
                row[f'msa_{key}'] = f"{mv:.6f}" if mv is not None else ""
                row[f'tmpl_{key}'] = f"{tv:.6f}" if tv is not None else ""
                if mv is not None and tv is not None:
                    row[f'delta_{key}'] = f"{tv - mv:.6f}"
                else:
                    row[f'delta_{key}'] = ""
            writer.writerow(row)

    print(f"\n  Wrote paired comparison CSV: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deep MSA vs template Boltz2 comparison")
    parser.add_argument("--msa-results", required=True)
    parser.add_argument("--template-results", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--ligand", default="LCA")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    # Load data
    labels = load_labels_csv(args.labels)
    print(f"Labels: {sum(labels.values())} binders, {sum(1 for v in labels.values() if not v)} non-binders")

    msa_data = merge(load_results_csv(args.msa_results), labels)
    tmpl_data = merge(load_results_csv(args.template_results), labels)

    shared_names = set(msa_data.keys()) & set(tmpl_data.keys())
    n_b = sum(1 for n in shared_names if msa_data[n]['is_binder'])
    n_nb = len(shared_names) - n_b

    print(f"\nMSA:      {len(msa_data)} matched variants")
    print(f"Template: {len(tmpl_data)} matched variants")
    print(f"Shared:   {len(shared_names)} ({n_b} binders, {n_nb} non-binders)")

    if len(shared_names) < 20:
        print("ERROR: too few shared predictions for meaningful comparison")
        sys.exit(1)

    # Run all analyses
    bootstrap_results = analysis_bootstrap_auc(msa_data, tmpl_data, shared_names)
    rank_results = analysis_rank_agreement(msa_data, tmpl_data, shared_names)
    analysis_confusion_overlap(msa_data, tmpl_data, shared_names)
    analysis_metric_correlations(msa_data, tmpl_data, shared_names)
    ensemble_results = analysis_ensemble(msa_data, tmpl_data, shared_names)
    analysis_distribution_shift(msa_data, tmpl_data, shared_names)
    analysis_variant_deltas(msa_data, tmpl_data, shared_names)

    # Output
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        write_paired_csv(msa_data, tmpl_data, shared_names, out_dir / 'paired_comparison.csv')

        print(f"\nGenerating figures...")
        make_figures(msa_data, tmpl_data, shared_names, bootstrap_results,
                     rank_results, ensemble_results, args.ligand, out_dir)

    print(f"\n{'='*80}")
    print(f"  DONE — Deep comparison complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
