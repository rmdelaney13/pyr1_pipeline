#!/usr/bin/env python3
"""
Select top N designs and copy their Boltz PDBs to a staging dir for MPNN redesign.

Supports three selection modes:
  1. Pure top-N by score (default, original behavior)
  2. Diverse: split N between top-score picks and Hamming-diverse picks
  3. Stratified: split by binding mode (normal vs flipped) with per-mode quotas,
     optionally sub-stratified by polar contact residue class.

Binding modes:
  normal  = OH at top of pocket, mediating conserved water
  flipped = COO at top of pocket, mediating conserved water

Uses pass_all, binary_binding_mode, binary_pocket_sequence, and
polar contact columns from the scored CSV (produced by analyze_boltz_output.py).

Usage:
    # Basic:
    python expansion_select.py \
        --scores /scratch/.../scores.csv \
        --boltz-dirs /scratch/.../output_ca_binary \
        --out-dir /scratch/.../round_1/selected_pdbs \
        --top-n 100

    # Diverse + stratified:
    python expansion_select.py \
        --scores /scratch/.../scores.csv \
        --boltz-dirs /scratch/.../output_ca_binary \
        --out-dir /scratch/.../round_1/selected_pdbs \
        --top-n 100 \
        --diverse --diverse-fraction 0.5 \
        --binding-mode-stratify --mode-quotas "flipped:50,normal:50"

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set


# Map legacy mode names (OH/COO) to current names (normal/flipped)
_MODE_REMAP = {'OH': 'normal', 'COO': 'flipped'}


def find_pdb_for_name(name: str, boltz_dirs: List[str]) -> Optional[Path]:
    """Locate the Boltz output PDB for a given prediction name.

    Searches: boltz_dir/boltz_results_{name}/predictions/{name}/{name}_model_0.pdb
    """
    for d in boltz_dirs:
        d = Path(d)
        # Direct path
        pdb = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.pdb"
        if pdb.exists():
            return pdb
        # Try CIF
        cif = d / f"boltz_results_{name}" / "predictions" / name / f"{name}_model_0.cif"
        if cif.exists():
            return cif
    return None


def hamming_dist(s1, s2):
    """Hamming distance between two equal-length strings."""
    return sum(a != b for a, b in zip(s1, s2))


def select_diverse(rows, n_total, diverse_frac, min_hamming=3,
                   min_diverse_score=None):
    """Select n_total designs: (1-diverse_frac)*n by score, rest by diversity.

    Diversity picks are chosen greedily to maximize minimum Hamming distance
    from already-selected pocket sequences. Uses binary_pocket_sequence from CSV.

    Args:
        min_diverse_score: if set, diversity candidates below this score
            threshold are excluded.
    """
    n_score = int(n_total * (1 - diverse_frac))
    n_diverse = n_total - n_score

    # Score picks (already sorted by score)
    score_picks = rows[:n_score]
    selected_set = set(r['name'] for r in score_picks)

    # Build pocket sequences for score picks
    selected_pockets = [r['binary_pocket_sequence'] for r in score_picks
                        if r.get('binary_pocket_sequence')]

    # Candidate pool for diversity picks
    candidates = []
    for r in rows[n_score:]:
        if r['name'] in selected_set:
            continue
        seq = r.get('binary_pocket_sequence', '')
        if not seq:
            continue
        if min_diverse_score is not None and r['_sort_score'] < min_diverse_score:
            continue
        candidates.append((r, seq))

    # Greedy diversity selection
    diverse_picks = []
    for _ in range(n_diverse):
        if not candidates:
            break

        best_row = None
        best_min_dist = -1
        best_idx = -1

        for ci, (r, seq) in enumerate(candidates):
            if not selected_pockets:
                min_d = 16  # max possible
            else:
                min_d = min(hamming_dist(seq, sp) for sp in selected_pockets)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_row = r
                best_idx = ci

        if best_row is None or best_min_dist < 1:
            break

        diverse_picks.append(best_row)
        _, seq = candidates[best_idx]
        selected_pockets.append(seq)
        candidates.pop(best_idx)

    print(f"  Diverse selection: {n_score} score picks + "
          f"{len(diverse_picks)} diversity picks"
          + (f" (min_score={min_diverse_score:.2f})" if min_diverse_score else ""))

    return score_picks + diverse_picks


def _classify_by_contact(rows, contact_column):
    """Split rows into {residue_int: [rows]} and no_contact list."""
    contact_buckets = defaultdict(list)
    no_contact = []
    for r in rows:
        res_val = r.get(contact_column, '')
        try:
            res = int(float(res_val))
            contact_buckets[res].append(r)
        except (ValueError, TypeError):
            no_contact.append(r)
    return contact_buckets, no_contact


def _print_contact_distribution(contact_buckets, no_contact, label):
    """Print contact residue distribution."""
    print(f"\n  {label} contact distribution (passing pool):")
    for res in sorted(contact_buckets):
        print(f"    res {res:>3d}: {len(contact_buckets[res])}")
    print(f"    no contact: {len(no_contact)}")


def select_contact_stratified(rows, n_total, contact_quotas,
                              diverse_frac, contact_column,
                              require_contact=False, mode_label='',
                              min_diverse_score=None):
    """Select designs stratified by polar contact residue (manual quotas).

    Guarantees minimum representation for each contact residue class,
    then fills remaining slots by score + sequence diversity from the
    full pool.
    """
    label = mode_label or contact_column
    contact_buckets, no_contact = _classify_by_contact(rows, contact_column)
    _print_contact_distribution(contact_buckets, no_contact, label)

    if require_contact:
        print(f"  --require-{label.lower()}-contact: excluding {len(no_contact)} "
              f"designs without direct contact")

    # Phase 1: fill reserved slots per contact residue (by score within each)
    selected = []
    selected_names = set()
    reserved_total = 0

    for res, quota in sorted(contact_quotas.items()):
        bucket = contact_buckets.get(res, [])
        picks = bucket[:min(quota, len(bucket))]
        selected.extend(picks)
        selected_names.update(r['name'] for r in picks)
        reserved_total += quota
        filled = len(picks)
        if filled < quota:
            print(f"    res {res:>3d}: reserved {filled}/{quota} (only {filled} available)")
        else:
            print(f"    res {res:>3d}: reserved {filled}/{quota}")

    # Phase 2: fill remaining slots from full pool via score + diversity
    n_remaining = n_total - len(selected)
    if n_remaining > 0:
        pool = []
        for r in rows:
            if r['name'] in selected_names:
                continue
            if require_contact:
                res_val = r.get(contact_column, '')
                try:
                    int(float(res_val))
                except (ValueError, TypeError):
                    continue
            pool.append(r)

        remaining_picks = select_diverse(pool, n_remaining, diverse_frac,
                                         min_diverse_score=min_diverse_score)
        selected.extend(remaining_picks)

    print(f"  {label} contact-stratified: {len(selected)} total "
          f"({reserved_total} reserved + {len(selected) - min(reserved_total, len(selected))} open)")

    return selected


def select_contact_adaptive(rows, n_total, diverse_frac, contact_column,
                            require_contact=False, mode_label='',
                            min_contact_classes=None,
                            min_diverse_score=None):
    """Select designs with adaptive equal allocation across contact classes.

    Divides n_total slots equally among all contact residue classes that
    have passing designs. Classes that can't fill their share donate surplus
    to remaining classes (iteratively). Within each class, selects by
    score + sequence diversity.
    """
    label = mode_label or contact_column
    contact_buckets, no_contact = _classify_by_contact(rows, contact_column)
    _print_contact_distribution(contact_buckets, no_contact, label)

    if require_contact:
        print(f"  --require-{label.lower()}-contact: excluding {len(no_contact)} "
              f"designs without direct contact")

    # Determine participating classes
    if min_contact_classes is not None:
        active_classes = {res for res in contact_buckets if res in min_contact_classes}
    else:
        active_classes = set(contact_buckets.keys())

    if not active_classes:
        print(f"  {label} adaptive: no contact classes found, falling back to score+diversity")
        return select_diverse(rows, n_total, diverse_frac,
                              min_diverse_score=min_diverse_score)

    # Iterative equal allocation with surplus redistribution
    remaining_slots = n_total
    class_allocations = {}  # {res: n_allocated}
    unfilled_classes = set(active_classes)

    while remaining_slots > 0 and unfilled_classes:
        per_class = remaining_slots // len(unfilled_classes)
        if per_class == 0:
            per_class = 1  # at least 1 per class in final pass

        surplus = 0
        newly_filled = set()
        for res in sorted(unfilled_classes):
            available = len(contact_buckets[res]) - class_allocations.get(res, 0)
            alloc = min(per_class, available)
            class_allocations[res] = class_allocations.get(res, 0) + alloc
            if alloc < per_class:
                surplus += per_class - alloc
                newly_filled.add(res)  # can't take more

        remaining_slots = surplus + (remaining_slots % len(unfilled_classes))
        unfilled_classes -= newly_filled

        # If no surplus was generated, we're done
        if surplus == 0:
            break

    # Select from each class
    selected = []
    selected_names = set()
    print(f"\n  {label} adaptive allocation ({len(active_classes)} classes):")

    for res in sorted(class_allocations, key=lambda r: -class_allocations.get(r, 0)):
        n_alloc = class_allocations[res]
        bucket = contact_buckets[res]
        if n_alloc <= 0:
            continue

        # Within-class selection: score + diversity
        if diverse_frac > 0 and n_alloc > 2:
            picks = select_diverse(bucket, n_alloc, diverse_frac,
                                   min_diverse_score=min_diverse_score)
        else:
            picks = bucket[:n_alloc]

        selected.extend(picks)
        selected_names.update(r['name'] for r in picks)
        avail = len(bucket)
        tag = " (ALL)" if len(picks) >= avail else ""
        print(f"    res {res:>3d}: {len(picks):>4d} / {avail:>4d} available{tag}")

    # Fill any remaining slots from overflow (non-active classes + no-contact)
    n_remaining = n_total - len(selected)
    if n_remaining > 0:
        overflow = []
        for res in sorted(contact_buckets):
            if res not in active_classes:
                overflow.extend(r for r in contact_buckets[res]
                                if r['name'] not in selected_names)
        if not require_contact:
            overflow.extend(r for r in no_contact
                            if r['name'] not in selected_names)
        overflow.sort(key=lambda r: -r['_sort_score'])
        extra = overflow[:n_remaining]
        selected.extend(extra)
        if extra:
            print(f"    overflow: {len(extra)} from non-active classes")

    print(f"  {label} adaptive total: {len(selected)}")
    return selected


def load_cluster_assignments(csv_path):
    """Load cluster CSV and return name->cluster_id mapping + metadata."""
    name_to_cluster = {}
    cluster_labels = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = int(row['cluster'])
            name_to_cluster[row['name']] = cid
            if cid not in cluster_labels:
                cluster_labels[cid] = row.get('cluster_label', '')
    return name_to_cluster, cluster_labels


def select_cluster_aware(rows, n_total, name_to_cluster, cluster_labels,
                         min_cluster_pass=3, small_quota=1,
                         diverse_frac=0.5, min_diverse_score=None,
                         require_oh=False):
    """Select designs with per-ligand-pose-cluster quotas.

    Every cluster gets representation. Larger/better clusters get more slots
    via iterative surplus redistribution (same pattern as contact-adaptive).

    Args:
        rows: Passing designs sorted by score descending.
        n_total: Total parent budget.
        name_to_cluster: dict mapping design name -> cluster ID.
        cluster_labels: dict mapping cluster ID -> label string.
        min_cluster_pass: Clusters with >= this many designs get full allocation.
        small_quota: Guaranteed slots for small clusters.
        diverse_frac: Fraction of within-cluster picks for Hamming diversity.
        min_diverse_score: Min score for diversity picks.
        require_oh: If True, kill clusters where no design has at least 1 OH satisfied.
    """
    # Group rows by cluster
    cluster_rows = defaultdict(list)
    unassigned = []
    for r in rows:
        cid = name_to_cluster.get(r['name'])
        if cid is not None:
            cluster_rows[cid].append(r)
        else:
            unassigned.append(r)

    if unassigned:
        print(f"  WARNING: {len(unassigned)} designs not in cluster CSV (new/unclustered)")

    # Filter clusters by OH satisfaction
    if require_oh:
        killed = set()
        for c, rr in cluster_rows.items():
            # Check if any design in this cluster has at least 1 OH satisfied
            has_oh = False
            for r in rr:
                mode = r.get('binary_binding_mode', '')
                n_oh_unsat = r.get('binary_n_oh_unsatisfied', '')
                try:
                    n_oh_unsat = int(n_oh_unsat)
                except (ValueError, TypeError):
                    continue
                # Normal mode: 2 OHs checked (7α, 12α). Need ≤1 unsat.
                # Flipped mode: 3 OHs checked (3α, 7α, 12α). Need ≤2 unsat.
                max_oh = 2 if mode == 'normal' else 3
                if n_oh_unsat < max_oh:
                    has_oh = True
                    break
            if not has_oh:
                killed.add(c)
        for c in killed:
            del cluster_rows[c]
        if killed:
            print(f"  --cluster-require-oh: killed {len(killed)} clusters "
                  f"with 0 OH-satisfied designs")

    n_clusters = len(cluster_rows)
    active = {c for c, rr in cluster_rows.items() if len(rr) >= min_cluster_pass}
    small = {c for c in cluster_rows if c not in active}

    print(f"\n  Cluster-aware selection: {n_clusters} clusters "
          f"({len(active)} active >= {min_cluster_pass}, {len(small)} small)")

    # Budget: small clusters get guaranteed small_quota, rest goes to active
    budget_small = min(len(small) * small_quota, n_total)
    budget_active = n_total - budget_small

    # Iterative equal allocation with surplus redistribution (active clusters)
    remaining = budget_active
    allocations = {}
    unfilled = set(active)

    while remaining > 0 and unfilled:
        per_cluster = max(1, remaining // len(unfilled))
        surplus = 0
        newly_filled = set()
        for c in sorted(unfilled):
            available = len(cluster_rows[c]) - allocations.get(c, 0)
            alloc = min(per_cluster, available)
            allocations[c] = allocations.get(c, 0) + alloc
            if alloc < per_cluster:
                surplus += per_cluster - alloc
                newly_filled.add(c)
        remaining = surplus + (remaining % len(unfilled))
        unfilled -= newly_filled
        if surplus == 0:
            break

    # Small clusters get their guaranteed quota
    for c in sorted(small):
        allocations[c] = min(small_quota, len(cluster_rows[c]))

    # Select within each cluster
    selected = []
    selected_names = set()
    sorted_clusters = sorted(allocations, key=lambda c: -allocations.get(c, 0))

    print(f"\n  Per-cluster allocations ({len(sorted_clusters)} clusters):")
    for i, c in enumerate(sorted_clusters):
        n_alloc = allocations[c]
        bucket = cluster_rows[c]
        label = cluster_labels.get(c, '')

        if n_alloc <= 0:
            continue

        # Within-cluster: score + diversity
        if diverse_frac > 0 and n_alloc > 2:
            picks = select_diverse(bucket, n_alloc, diverse_frac,
                                   min_diverse_score=min_diverse_score)
        else:
            picks = bucket[:n_alloc]

        selected.extend(picks)
        selected_names.update(r['name'] for r in picks)

        # Print details for top 30 clusters
        if i < 30:
            modes = Counter(r.get('binary_binding_mode', '?') for r in picks)
            mode_str = '/'.join(f'{m}:{v}' for m, v in modes.most_common())
            oh_unsat = [int(r.get('binary_n_oh_unsatisfied', 99)) for r in picks]
            n_oh0 = sum(1 for x in oh_unsat if x == 0)
            print(f"    cluster {c:>3d}: {len(picks):>3d}/{len(bucket):>3d}  "
                  f"{label:25s} {mode_str:20s} oh0={n_oh0}")

    remaining_shown = len(sorted_clusters) - 30
    if remaining_shown > 0:
        print(f"    ... and {remaining_shown} more clusters")

    # Fill any leftover from unassigned pool
    n_remaining = n_total - len(selected)
    if n_remaining > 0 and unassigned:
        extra = [r for r in unassigned if r['name'] not in selected_names][:n_remaining]
        selected.extend(extra)
        if extra:
            print(f"    unassigned overflow: {len(extra)}")

    n_clusters_repr = len({name_to_cluster[r['name']]
                           for r in selected if r['name'] in name_to_cluster})
    print(f"\n  Cluster-aware total: {len(selected)} from {n_clusters_repr} clusters")
    return selected


def parse_mode_quotas(quota_str):
    """Parse 'flipped:100,normal:100' into dict {'flipped': 100, 'normal': 100}.

    Also accepts legacy names (COO->flipped, OH->normal).
    """
    quotas = {}
    for part in quota_str.split(','):
        mode, count = part.strip().split(':')
        mode = mode.strip()
        # Remap legacy names
        mode = _MODE_REMAP.get(mode.upper(), mode.lower())
        quotas[mode] = int(count.strip())
    return quotas


def _contact_column_for_mode(mode):
    """Return the appropriate contact CSV column for a binding mode.

    Both modes use binary_oh_contact_res as the diversity key:
      normal:  core sterol OH is buried in pocket, contact residue = orientation.
      flipped: both sterol OHs are buried in pocket, contact residue = orientation.

    The COO contact (binary_coo_contact_res) is NOT a good diversity key because
    in flipped mode both COO oxygens are at the gate — the contact residue is just
    whatever's nearest to the top of the pocket, not informative about ligand pose.
    """
    return 'binary_oh_contact_res'


def select_stratified(rows, quotas, diverse, diverse_frac,
                      oh_contact_quotas=None, require_oh_contact=False,
                      coo_contact_quotas=None, require_coo_contact=False,
                      adaptive_contact=False, adaptive_min_classes=None,
                      min_diverse_score=None):
    """Select designs stratified by binding mode with per-mode quotas.

    Within each mode, applies score-only or score+diversity selection.
    For normal/flipped modes, optionally sub-stratifies by polar contact
    residue to ensure underrepresented binding poses get parent slots.

    Contact column per mode:
      normal:  binary_oh_contact_res (core sterol OH buried in pocket)
      flipped: binary_coo_contact_res (carboxylate buried in pocket)
    """
    # Classify all rows from CSV column
    mode_buckets = defaultdict(list)
    for r in rows:
        mode = r.get('binary_binding_mode', 'unknown')
        r['_binding_mode'] = mode
        mode_buckets[mode].append(r)

    print(f"\n  Binding mode distribution:")
    for mode in ['normal', 'flipped', 'unknown']:
        print(f"    {mode}: {len(mode_buckets[mode])}")

    # First pass: fill each mode's quota
    selected = []
    remaining_quota = 0
    modes_with_surplus = []

    for mode, quota in quotas.items():
        bucket = mode_buckets.get(mode, [])
        if not bucket:
            remaining_quota += quota
            print(f"    {mode}: filled 0/{quota} (empty)")
            continue

        _require = require_oh_contact if mode == 'normal' else require_coo_contact
        _manual_quotas = oh_contact_quotas if mode == 'normal' else coo_contact_quotas
        _min_classes = adaptive_min_classes.get(mode) if adaptive_min_classes else None
        _contact_col = _contact_column_for_mode(mode)

        if adaptive_contact and mode in ('normal', 'flipped'):
            picks = select_contact_adaptive(
                bucket, min(quota, len(bucket)),
                diverse_frac,
                contact_column=_contact_col,
                require_contact=_require,
                mode_label=mode,
                min_contact_classes=_min_classes,
                min_diverse_score=min_diverse_score)
        elif _manual_quotas and mode in ('normal', 'flipped'):
            picks = select_contact_stratified(
                bucket, min(quota, len(bucket)),
                _manual_quotas, diverse_frac,
                contact_column=_contact_col,
                require_contact=_require,
                mode_label=mode,
                min_diverse_score=min_diverse_score)
        elif diverse:
            picks = select_diverse(
                bucket, min(quota, len(bucket)), diverse_frac,
                min_diverse_score=min_diverse_score)
        else:
            picks = bucket[:min(quota, len(bucket))]

        selected.extend(picks)
        shortfall = quota - len(picks)
        if shortfall > 0:
            remaining_quota += shortfall
            print(f"    {mode}: filled {len(picks)}/{quota} "
                  f"(shortfall {shortfall})")
        else:
            modes_with_surplus.append(mode)
            print(f"    {mode}: filled {len(picks)}/{quota}")

    # Redistribute shortfall to modes with surplus
    if remaining_quota > 0 and modes_with_surplus:
        selected_names = set(r['name'] for r in selected)
        extra_per_mode = remaining_quota // len(modes_with_surplus)
        leftover = remaining_quota % len(modes_with_surplus)

        for i, mode in enumerate(modes_with_surplus):
            n_extra = extra_per_mode + (1 if i < leftover else 0)
            if n_extra == 0:
                continue
            bucket = [r for r in mode_buckets[mode]
                      if r['name'] not in selected_names]
            extra = bucket[:n_extra]
            selected.extend(extra)
            if extra:
                print(f"    {mode}: +{len(extra)} redistributed")

    # Also fill from 'unknown' if still short
    total_quota = sum(quotas.values())
    if len(selected) < total_quota and mode_buckets.get('unknown'):
        selected_names = set(r['name'] for r in selected)
        n_need = total_quota - len(selected)
        unknown = [r for r in mode_buckets['unknown']
                   if r['name'] not in selected_names]
        extra = unknown[:n_need]
        selected.extend(extra)
        if extra:
            print(f"    unknown: +{len(extra)} to fill remaining quota")

    return selected


def dedup_hamming(selected, min_hamming=2):
    """Remove designs whose pocket sequence is within min_hamming of an earlier pick.

    Preserves order (earlier = higher priority). Returns (kept, n_dropped).
    """
    kept = []
    kept_seqs = []
    dropped = 0
    for r in selected:
        seq = r.get('binary_pocket_sequence', '')
        if not seq:
            kept.append(r)
            continue
        if any(hamming_dist(seq, s) < min_hamming for s in kept_seqs):
            dropped += 1
            continue
        kept.append(r)
        kept_seqs.append(seq)
    return kept, dropped


def print_diversity_summary(selected_rows):
    """Print diversity statistics for selected designs."""
    mode_counts = Counter(r.get('binary_binding_mode', r.get('_binding_mode', 'unknown'))
                          for r in selected_rows)
    mode_str = ', '.join(f"{m}: {mode_counts[m]}" for m in ['normal', 'flipped', 'unknown']
                         if mode_counts.get(m, 0))
    print(f"  Modes: {mode_str}")

    # Per-mode OH contact breakdown (OH contacts are the diversity key for both modes)
    for mode, label in [('normal', 'Normal OH contacts'), ('flipped', 'Flipped OH contacts')]:
        mode_rows = [r for r in selected_rows
                     if r.get('binary_binding_mode', r.get('_binding_mode')) == mode]
        if not mode_rows:
            continue
        contact_counts = Counter()
        for r in mode_rows:
            res_val = r.get('binary_oh_contact_res', '')
            try:
                contact_counts[int(float(res_val))] += 1
            except (ValueError, TypeError):
                contact_counts['none'] += 1
        contact_str = ', '.join(
            f"{k}: {v}" for k, v in sorted(
                contact_counts.items(),
                key=lambda x: (-x[1], str(x[0]))))
        print(f"  {label}: {contact_str}")

    pocket_seqs = [r.get('binary_pocket_sequence', '') for r in selected_rows
                   if r.get('binary_pocket_sequence')]
    unique_seqs = set(pocket_seqs)
    print(f"  Unique pocket sequences: {len(unique_seqs)}")

    if len(pocket_seqs) >= 2:
        # Compute mean pairwise Hamming (sample if large)
        if len(pocket_seqs) > 200:
            sample = random.sample(pocket_seqs, 200)
        else:
            sample = pocket_seqs
        dists = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                dists.append(hamming_dist(sample[i], sample[j]))
        if dists:
            mean_h = sum(dists) / len(dists)
            min_h = min(dists)
            print(f"  Mean pairwise Hamming: {mean_h:.1f}, Min: {min_h}")

    # Per-mode score summary
    for mode in ['normal', 'flipped']:
        mode_rows = [r for r in selected_rows
                     if r.get('binary_binding_mode', r.get('_binding_mode')) == mode]
        if mode_rows:
            scores = [r['_sort_score'] for r in mode_rows]
            print(f"  {mode} scores: min={min(scores):.2f}, "
                  f"mean={sum(scores)/len(scores):.2f}, max={max(scores):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Select top N designs and copy PDBs for MPNN redesign")
    parser.add_argument("--scores", required=True,
                        help="Scores CSV from analyze_boltz_output.py")
    parser.add_argument("--boltz-dirs", nargs='+', action='append', required=True,
                        help="Boltz output directories to search for PDBs "
                             "(can be repeated: --boltz-dirs A B --boltz-dirs C)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for selected PDBs")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top designs to select (default: 100)")
    parser.add_argument("--score-column", default="binary_total_score",
                        help="Column to rank by (default: binary_total_score)")
    parser.add_argument("--diverse", action="store_true",
                        help="Use diverse parent selection: split top-N between "
                             "score picks and Hamming-diverse picks")
    parser.add_argument("--diverse-fraction", type=float, default=0.5,
                        help="Fraction of top-N to fill with diversity picks "
                             "(default: 0.5 = 50/50 split)")
    parser.add_argument("--min-diverse-score", type=float, default=None,
                        help="Minimum score threshold for diversity picks. "
                             "Prevents the Hamming search from selecting low-quality "
                             "designs just because they are sequence-different. "
                             "E.g. 1.5 excludes designs with total_score < 1.5 "
                             "from the diversity candidate pool.")
    parser.add_argument("--binding-mode-stratify", action="store_true",
                        help="Stratify selection by binding mode (normal vs flipped). "
                             "Uses binary_binding_mode column from CSV.")
    parser.add_argument("--mode-quotas", default=None,
                        help="Per-mode quotas, e.g. 'flipped:100,normal:100'. "
                             "Also accepts legacy names COO/OH. "
                             "Underfilled quotas redistribute to other modes. "
                             "Requires --binding-mode-stratify.")
    parser.add_argument("--exclude-pocket-aa", nargs='*', default=[],
                        help="Exclude designs with specific AA at pocket positions. "
                             "Format: POS:AA (Boltz numbering), e.g. '159:D 117:P'. "
                             "Uses binary_pocket_sequence column.")
    parser.add_argument("--oh-contact-quotas", default=None,
                        help="Min slots per OH contact residue for normal mode, "
                             "e.g. '92:50,120:10,160:5'. "
                             "Guarantees representation for rare binding poses. "
                             "Requires --binding-mode-stratify.")
    parser.add_argument("--require-oh-contact", action="store_true",
                        help="Exclude normal-mode designs without a direct protein "
                             "polar contact to the sterol hydroxyl "
                             "(binary_oh_contact_res must be set).")
    parser.add_argument("--coo-contact-quotas", default=None,
                        help="Min slots per COO contact residue for flipped mode, "
                             "e.g. '94:50,167:30,59:30'. "
                             "Uses binary_coo_contact_res column. "
                             "Requires --binding-mode-stratify.")
    parser.add_argument("--require-coo-contact", action="store_true",
                        help="Exclude flipped-mode designs without a direct protein "
                             "polar contact to the carboxylate "
                             "(binary_coo_contact_res must be set).")
    parser.add_argument("--adaptive-contact-balance", action="store_true",
                        help="Adaptively divide each mode's slots equally across all "
                             "contact residue classes. Rare classes get ALL their "
                             "designs, surplus redistributes to larger classes. "
                             "Replaces manual --oh/coo-contact-quotas. "
                             "Requires --binding-mode-stratify.")
    parser.add_argument("--adaptive-normal-classes", default=None,
                        help="Restrict adaptive normal-mode allocation to these OH "
                             "contact residues, e.g. '92,117,120,160,110,122'. "
                             "Others go to overflow. Only with --adaptive-contact-balance.")
    parser.add_argument("--adaptive-flipped-classes", default=None,
                        help="Restrict adaptive flipped-mode allocation to these COO "
                             "contact residues, e.g. '94,167,59,163,120'. "
                             "Others go to overflow. Only with --adaptive-contact-balance.")
    # Keep legacy arg names working (hidden)
    parser.add_argument("--adaptive-oh-classes", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--adaptive-coo-classes", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cluster-csv", default=None,
                        help="Path to cluster CSV (e.g. clusters_2.0A.csv). "
                             "Enables per-pose-cluster quota selection: each "
                             "ligand pose cluster gets a share of the parent budget.")
    parser.add_argument("--cluster-min-pass", type=int, default=3,
                        help="Min designs in a cluster for full allocation "
                             "(smaller clusters get --cluster-small-quota). Default: 3")
    parser.add_argument("--cluster-small-quota", type=int, default=1,
                        help="Guaranteed slots for below-threshold clusters. Default: 1")
    parser.add_argument("--cluster-require-oh", action="store_true",
                        help="Only select from clusters that have at least 1 design "
                             "with at least 1 OH satisfied (n_oh_unsatisfied < max). "
                             "Kills clusters where no sequence satisfies any hydroxyl.")
    parser.add_argument("--hamming-dedup", type=int, default=None, metavar='MIN_DIST',
                        help="Post-selection global Hamming deduplication. Remove "
                             "designs whose pocket sequence is within MIN_DIST of "
                             "an earlier (higher-priority) pick. E.g. --hamming-dedup 2 "
                             "removes near-duplicate sequences across contact classes.")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten boltz_dirs: action='append' + nargs='+' gives list of lists
    boltz_dirs = [d for group in args.boltz_dirs for d in group]

    print(f"Searching {len(boltz_dirs)} Boltz output directories:")
    for d in boltz_dirs:
        exists = Path(d).is_dir()
        print(f"  {'OK' if exists else 'MISSING'}: {d}")

    # Read and sort scores
    rows = []
    with open(args.scores) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_val = row.get(args.score_column)
            if score_val is None or score_val == '':
                continue
            try:
                row['_sort_score'] = float(score_val)
            except ValueError:
                continue
            # Remap legacy binding mode names (OH->normal, COO->flipped)
            mode = row.get('binary_binding_mode', '')
            if mode in _MODE_REMAP:
                row['binary_binding_mode'] = _MODE_REMAP[mode]
            rows.append(row)

    rows.sort(key=lambda r: r['_sort_score'], reverse=True)
    print(f"\nLoaded {len(rows)} scored designs")

    # Filter to passing designs only
    has_pass_all = any(r.get('pass_all') is not None and r.get('pass_all') != '' for r in rows)
    if has_pass_all:
        before = len(rows)
        rows = [r for r in rows if r.get('pass_all') == '1' or r.get('pass_all') == 1]
        print(f"  After pass_all gate: {before} -> {len(rows)}")
    else:
        print("  WARNING: pass_all column not found, no gate filtering applied")

    # Exclude designs with forbidden AA at specific pocket positions
    if args.exclude_pocket_aa:
        POCKET_POSITIONS = [59, 81, 83, 92, 94, 108, 110, 117, 120, 122, 141, 159, 160, 163, 164, 167]
        pos_to_idx = {p: i for i, p in enumerate(POCKET_POSITIONS)}
        exclusions = []
        for spec in args.exclude_pocket_aa:
            pos_str, aa = spec.split(':')
            pos = int(pos_str)
            if pos not in pos_to_idx:
                print(f"  WARNING: position {pos} not in pocket positions, skipping")
                continue
            exclusions.append((pos_to_idx[pos], aa.upper(), pos))

        before = len(rows)
        filtered = []
        for r in rows:
            seq = r.get('binary_pocket_sequence', '')
            exclude = False
            for idx, aa, pos in exclusions:
                if idx < len(seq) and seq[idx] == aa:
                    exclude = True
                    break
            if not exclude:
                filtered.append(r)
        rows = filtered
        exc_str = ', '.join(f'{pos}:{aa}' for _, aa, pos in exclusions)
        print(f"  After --exclude-pocket-aa ({exc_str}): {before} -> {len(rows)}")

    # NOTE: Rows stay sorted by score only. The unsat_penalty in
    # analyze_boltz_output.py already penalizes unsatisfied polars in
    # binary_total_score, so no secondary sort by n_polar_unsatisfied is needed.

    # Parse contact quotas
    def _parse_contact_quotas(spec):
        quotas = {}
        for part in spec.split(','):
            res_str, count = part.strip().split(':')
            count = count.strip()
            quotas[int(res_str.strip())] = 99999 if count.upper() == 'ALL' else int(count)
        return quotas

    oh_contact_quotas = _parse_contact_quotas(args.oh_contact_quotas) \
        if args.oh_contact_quotas else None
    coo_contact_quotas = _parse_contact_quotas(args.coo_contact_quotas) \
        if args.coo_contact_quotas else None

    # Parse adaptive class restrictions (support both new and legacy arg names)
    adaptive_min_classes = None
    if args.adaptive_contact_balance:
        adaptive_min_classes = {}
        normal_classes = args.adaptive_normal_classes or args.adaptive_oh_classes
        flipped_classes = args.adaptive_flipped_classes or args.adaptive_coo_classes
        if normal_classes:
            adaptive_min_classes['normal'] = {int(x) for x in normal_classes.split(',')}
        if flipped_classes:
            adaptive_min_classes['flipped'] = {int(x) for x in flipped_classes.split(',')}

    # Load cluster assignments if provided
    name_to_cluster = None
    cluster_labels = None
    if args.cluster_csv:
        name_to_cluster, cluster_labels = load_cluster_assignments(args.cluster_csv)
        print(f"\nLoaded {len(set(name_to_cluster.values()))} clusters from {args.cluster_csv}")
        # Count how many scored rows have cluster assignments
        n_matched = sum(1 for r in rows if r['name'] in name_to_cluster)
        print(f"  Matched: {n_matched}/{len(rows)} passing designs")

    # Select designs
    if args.cluster_csv:
        print(f"\nCluster-aware selection: {args.top_n} parents across "
              f"{len(set(name_to_cluster.values()))} pose clusters")
        top_rows = select_cluster_aware(
            rows, args.top_n, name_to_cluster, cluster_labels,
            min_cluster_pass=args.cluster_min_pass,
            small_quota=args.cluster_small_quota,
            diverse_frac=args.diverse_fraction if args.diverse else 0,
            min_diverse_score=args.min_diverse_score,
            require_oh=args.cluster_require_oh)
    elif args.binding_mode_stratify:
        if not args.mode_quotas:
            parser.error("--binding-mode-stratify requires --mode-quotas")
        quotas = parse_mode_quotas(args.mode_quotas)
        print(f"\nBinding-mode stratified selection: {quotas}")
        if args.adaptive_contact_balance:
            print(f"Adaptive contact balance: ON")
            print(f"  normal mode -> binary_oh_contact_res")
            print(f"  flipped mode -> binary_coo_contact_res")
            if adaptive_min_classes:
                for m, cls in sorted(adaptive_min_classes.items()):
                    print(f"  {m} classes: {sorted(cls)}")
        elif oh_contact_quotas:
            print(f"Normal OH contact quotas: {oh_contact_quotas}")
        if coo_contact_quotas and not args.adaptive_contact_balance:
            print(f"Flipped COO contact quotas: {coo_contact_quotas}")
        if args.min_diverse_score is not None:
            print(f"Min diverse score: {args.min_diverse_score}")
        top_rows = select_stratified(
            rows, quotas, args.diverse, args.diverse_fraction,
            oh_contact_quotas=oh_contact_quotas,
            require_oh_contact=args.require_oh_contact,
            coo_contact_quotas=coo_contact_quotas,
            require_coo_contact=args.require_coo_contact,
            adaptive_contact=args.adaptive_contact_balance,
            adaptive_min_classes=adaptive_min_classes,
            min_diverse_score=args.min_diverse_score)
    elif args.diverse:
        top_rows = select_diverse(
            rows, args.top_n, args.diverse_fraction,
            min_diverse_score=args.min_diverse_score)
    else:
        top_rows = rows[:args.top_n]

    # Global Hamming deduplication across contact classes
    if args.hamming_dedup is not None and args.hamming_dedup > 0:
        before_dedup = len(top_rows)
        top_rows, n_dropped = dedup_hamming(top_rows, min_hamming=args.hamming_dedup)
        print(f"\nHamming dedup (min_dist={args.hamming_dedup}): "
              f"{before_dedup} -> {len(top_rows)} ({n_dropped} near-duplicates removed)")

    print(f"\nSelected {len(top_rows)} designs")
    if top_rows:
        scores = [r['_sort_score'] for r in top_rows]
        print(f"  Score range: {max(scores):.4f} - {min(scores):.4f}")

    # Diversity summary
    print_diversity_summary(top_rows)

    # Copy PDBs
    copied = 0
    missing = 0
    missing_names = []
    manifest_lines = []

    for row in top_rows:
        name = row['name']
        pdb_path = find_pdb_for_name(name, boltz_dirs)
        if pdb_path is None:
            missing += 1
            missing_names.append(name)
            continue

        dest = out_dir / pdb_path.name
        shutil.copy2(pdb_path, dest)
        manifest_lines.append(str(dest))
        copied += 1

    # Write manifest
    manifest_path = out_dir.parent / "selected_manifest.txt"
    with open(manifest_path, 'w') as f:
        for line in manifest_lines:
            f.write(line + '\n')

    print(f"\nCopied {copied} PDBs to {out_dir}")
    if missing:
        print(f"  ({missing} PDBs not found)", file=sys.stderr)
        # Show first few missing names for debugging
        show = missing_names[:5]
        print(f"  First missing: {', '.join(show)}" +
              (f" (+ {missing - 5} more)" if missing > 5 else ""),
              file=sys.stderr)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
