#!/usr/bin/env python3
"""
Pocket sequence evolution heatmap + diversity-aware design selection.

Part 1: Visualize how AA frequencies at 16 pocket positions shift across
        expansion rounds (4x4 stacked bar heatmap).
Part 2: Cluster gated designs by pocket Hamming distance, then round-robin
        select top-N designs maximizing diversity for Twist ordering.

Usage:
    # Heatmap only
    python scripts/analyze_pocket_evolution.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --initial-csv-dir /scratch/alpine/ryde3462/boltz_bile_acids/csvs \
        --ligands ca cdca dca --plot

    # Diversity selection (after filtering)
    python scripts/analyze_pocket_evolution.py \
        --expansion-root /scratch/alpine/ryde3462/expansion/ligandmpnn \
        --initial-csv-dir /scratch/alpine/ryde3462/boltz_bile_acids/csvs \
        --ligands ca cdca dca \
        --select-diverse 165 --min-hamming 3 --plot
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Import from existing pipeline scripts
sys.path.insert(0, str(Path(__file__).parent))
from expansion_pocket_analysis import (
    WT_PYR1_SEQUENCE,
    POCKET_POSITIONS,
    build_signature_lookup,
    pocket_from_signature,
    wt_at_position,
)

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_ROOT = "/scratch/alpine/ryde3462/expansion/ligandmpnn"
DEFAULT_INITIAL = "/scratch/alpine/ryde3462/boltz_bile_acids/csvs"
DEFAULT_LIGANDS = ["ca", "cdca", "dca"]

# Consistent AA color palette (hydrophobic=warm, polar=cool, charged=bright)
AA_COLORS = {
    'A': '#C8C8C8', 'G': '#D0D0D0',  # small, grey
    'V': '#E8B960', 'L': '#D4A030', 'I': '#C09020', 'M': '#B08010',  # hydrophobic, amber
    'F': '#E06030', 'W': '#D04020', 'Y': '#C05040', 'P': '#A08060',  # aromatic/Pro, red-orange
    'S': '#60B0E0', 'T': '#5098D0', 'C': '#40C080', 'N': '#70C0A0', 'Q': '#60B090',  # polar, blue-green
    'D': '#E04040', 'E': '#D03030',  # acidic, red
    'K': '#4060E0', 'R': '#3050D0', 'H': '#6080C0',  # basic, blue
}


def parse_round_from_name(name: str) -> int:
    """Extract round from design name. No _exp_r → 0."""
    import re
    m = re.search(r"_exp_r(\d+)_", name)
    return int(m.group(1)) if m else 0


def pocket_string(pocket_dict):
    """Convert pocket dict to 16-char string in canonical order."""
    return ''.join(pocket_dict.get(p, '?') for p in POCKET_POSITIONS)


def hamming_distance(seq1, seq2):
    """Hamming distance between two equal-length strings."""
    return sum(a != b for a, b in zip(seq1, seq2))


def shannon_entropy(counter):
    """Shannon entropy (bits) from a Counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counter.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_round_frequencies(lig, root, initial_csv_dir):
    """Compute AA frequency at each pocket position for each round.

    Returns: dict[round_num] → dict[position] → Counter(AA → count)
    """
    lig_dir = Path(root) / lig
    lookup = build_signature_lookup(initial_csv_dir, root, lig)

    round_freqs = {}

    for round_dir in sorted(lig_dir.glob("round_*")):
        rn_str = round_dir.name.replace("round_", "")
        if not rn_str.isdigit():
            continue
        rn = int(rn_str)

        # Load scores for this round
        scores_file = None
        for fname in ["new_scores.csv", "scores.csv"]:
            candidate = round_dir / fname
            if candidate.exists():
                scores_file = candidate
                break
        if scores_file is None:
            continue

        position_freqs = {pos: Counter() for pos in POCKET_POSITIONS}
        mapped = 0

        with open(scores_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "")
                sig = lookup.get(name)
                if sig is None:
                    continue
                pocket = pocket_from_signature(sig)
                for pos in POCKET_POSITIONS:
                    position_freqs[pos][pocket[pos]] += 1
                mapped += 1

        if mapped > 0:
            round_freqs[rn] = {"freqs": position_freqs, "n": mapped}

    return round_freqs


def plot_pocket_evolution(lig, round_freqs, out_path):
    """4x4 stacked bar chart: one subplot per pocket position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rounds = sorted(round_freqs.keys())
    if len(rounds) < 2:
        print(f"  {lig.upper()}: only {len(rounds)} round(s), skipping heatmap")
        return

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    fig.suptitle(f"{lig.upper()} — Pocket AA Evolution Across Rounds",
                 fontsize=14, fontweight="bold", y=0.98)

    # Collect all AAs seen across all positions/rounds
    all_aas = set()
    for rn_data in round_freqs.values():
        for pos in POCKET_POSITIONS:
            all_aas.update(rn_data["freqs"][pos].keys())

    for idx, (ax, pos) in enumerate(zip(axes.flat, POCKET_POSITIONS)):
        wt = wt_at_position(pos)

        # Collect all AAs at this position across rounds
        pos_aas = set()
        for rn_data in round_freqs.values():
            pos_aas.update(rn_data["freqs"][pos].keys())

        # Sort: WT first, then by total frequency descending
        total_counts = Counter()
        for rn_data in round_freqs.values():
            total_counts.update(rn_data["freqs"][pos])
        sorted_aas = sorted(pos_aas, key=lambda aa: (-1e9 if aa == wt else 0, -total_counts[aa]))

        # Build stacked bar data
        bottoms = np.zeros(len(rounds))
        for aa in sorted_aas:
            heights = []
            for rn in rounds:
                total = round_freqs[rn]["n"]
                count = round_freqs[rn]["freqs"][pos].get(aa, 0)
                heights.append(count / total if total > 0 else 0)
            heights = np.array(heights)
            color = AA_COLORS.get(aa, '#808080')
            if aa == wt:
                color = '#D0D0D0'  # WT always grey
            ax.bar(rounds, heights, bottom=bottoms, color=color,
                   edgecolor='white', linewidth=0.3, label=aa, width=0.7)
            bottoms += heights

        # Entropy annotation
        entropies = []
        for rn in rounds:
            entropies.append(shannon_entropy(round_freqs[rn]["freqs"][pos]))
        ax.set_title(f"Pos {pos} (WT={wt})", fontsize=9, fontweight="bold")

        # Annotate last entropy
        if entropies:
            e0 = entropies[0]
            elast = entropies[-1]
            delta = elast - e0
            symbol = "+" if delta > 0 else ""
            ax.text(0.97, 0.97, f"H={elast:.1f} ({symbol}{delta:.1f})",
                    transform=ax.transAxes, fontsize=7, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_ylim(0, 1.05)
        ax.set_xticks(rounds)
        ax.set_xticklabels([str(r) for r in rounds], fontsize=7)
        if idx % 4 == 0:
            ax.set_ylabel("Frequency", fontsize=8)
        if idx >= 12:
            ax.set_xlabel("Round", fontsize=8)

        # Legend: only show top 5 AAs
        top5 = sorted_aas[:5]
        handles = [Patch(facecolor=AA_COLORS.get(aa, '#808080') if aa != wt else '#D0D0D0',
                         label=aa) for aa in top5]
        if len(sorted_aas) > 5:
            handles.append(Patch(facecolor='#808080', label=f"+{len(sorted_aas)-5}"))
        ax.legend(handles=handles, fontsize=6, loc="upper left", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Diversity-aware selection
# ═══════════════════════════════════════════════════════════════════════════

def greedy_hamming_cluster(pocket_seqs, scores, min_dist=3):
    """Greedy centroid clustering by pocket Hamming distance.

    Args:
        pocket_seqs: list of 16-char pocket strings
        scores: list of scores (higher = better), same order
        min_dist: minimum Hamming distance to form new cluster

    Returns:
        cluster_ids: list of cluster assignments (0-indexed)
        centroids: list of (index, pocket_seq) for each cluster centroid
    """
    # Sort by score descending
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    cluster_ids = [-1] * len(pocket_seqs)
    centroids = []  # (original_index, pocket_seq)

    for idx in order:
        seq = pocket_seqs[idx]
        if not centroids:
            cluster_ids[idx] = 0
            centroids.append((idx, seq))
            continue

        # Find nearest centroid
        min_d = min(hamming_distance(seq, c_seq) for _, c_seq in centroids)
        if min_d >= min_dist:
            # New cluster
            cluster_ids[idx] = len(centroids)
            centroids.append((idx, seq))
        else:
            # Assign to nearest
            best_c = min(range(len(centroids)),
                         key=lambda c: hamming_distance(seq, centroids[c][1]))
            cluster_ids[idx] = best_c

    return cluster_ids, centroids


def diversity_select(df, pocket_col, score_col, n_select, min_hamming=3):
    """Round-robin selection across Hamming clusters for diversity.

    Args:
        df: DataFrame with pocket sequences and scores
        pocket_col: column name with 16-char pocket sequence
        score_col: column name with score (higher = better)
        n_select: number of designs to select
        min_hamming: minimum Hamming distance for clustering

    Returns:
        selected_indices: list of DataFrame indices selected
        cluster_ids: cluster assignment for all rows
    """
    pocket_seqs = df[pocket_col].tolist()
    scores = df[score_col].tolist()

    cluster_ids, centroids = greedy_hamming_cluster(pocket_seqs, scores, min_hamming)
    n_clusters = len(centroids)

    print(f"    Formed {n_clusters} clusters (min Hamming = {min_hamming})")
    for c_idx, (_, c_seq) in enumerate(centroids):
        c_size = sum(1 for cid in cluster_ids if cid == c_idx)
        print(f"      Cluster {c_idx}: {c_size} designs, centroid = {c_seq}")

    # Round-robin selection: iterate over clusters, pick best unselected from each
    # Sort each cluster's members by score descending
    cluster_members = {}
    for i, cid in enumerate(cluster_ids):
        if cid not in cluster_members:
            cluster_members[cid] = []
        cluster_members[cid].append((scores[i], i))

    for cid in cluster_members:
        cluster_members[cid].sort(reverse=True)

    selected = []
    selected_set = set()
    cluster_pointers = {cid: 0 for cid in range(n_clusters)}

    while len(selected) < n_select:
        added_this_round = False
        for cid in range(n_clusters):
            if len(selected) >= n_select:
                break
            members = cluster_members.get(cid, [])
            ptr = cluster_pointers[cid]
            while ptr < len(members):
                _, idx = members[ptr]
                ptr += 1
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    cluster_pointers[cid] = ptr
                    added_this_round = True
                    break
            cluster_pointers[cid] = ptr

        if not added_this_round:
            break  # exhausted all clusters

    return selected, cluster_ids


def run_diversity_selection(lig, root, initial_csv_dir, n_select, min_hamming):
    """Load filtered designs, cluster, and select diverse top-N."""
    filtered_dir = Path(root) / "filtered"
    filtered_csv = filtered_dir / f"top100_{lig}.csv"

    if not filtered_csv.exists():
        # Try larger top-N files
        for n in [200, 500]:
            alt = filtered_dir / f"top{n}_{lig}.csv"
            if alt.exists():
                filtered_csv = alt
                break

    if not filtered_csv.exists():
        print(f"  {lig.upper()}: no filtered CSV found in {filtered_dir}, skipping")
        return None

    df = pd.read_csv(filtered_csv)
    print(f"  {lig.upper()}: loaded {len(df)} filtered designs from {filtered_csv.name}")

    # Get pocket sequences
    lookup = build_signature_lookup(initial_csv_dir, root, lig)

    pocket_seqs = []
    for _, row in df.iterrows():
        name = row.get("name", "")
        # Try sequence column first
        if "sequence" in df.columns and pd.notna(row.get("sequence")):
            full_seq = str(row["sequence"])
            pocket = ''.join(full_seq[p - 1] for p in POCKET_POSITIONS
                             if p - 1 < len(full_seq))
        else:
            sig = lookup.get(name, "")
            pocket_dict = pocket_from_signature(sig)
            pocket = pocket_string(pocket_dict)
        pocket_seqs.append(pocket)

    df["pocket_seq"] = pocket_seqs

    # Score column
    score_col = "composite_zscore"
    if score_col not in df.columns:
        score_col = "binary_total_score"
    if score_col not in df.columns:
        print(f"  {lig.upper()}: no score column found, skipping diversity selection")
        return None

    # Fill NaN scores with -inf
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(-999)

    selected_indices, cluster_ids = diversity_select(
        df, "pocket_seq", score_col, n_select, min_hamming
    )

    df["cluster_id"] = cluster_ids

    # Create output
    selected_df = df.iloc[selected_indices].copy()
    selected_df["diversity_rank"] = range(1, len(selected_df) + 1)

    out_csv = filtered_dir / f"diverse_{lig}.csv"
    selected_df.to_csv(out_csv, index=False)
    print(f"  {lig.upper()}: selected {len(selected_df)} diverse designs → {out_csv}")

    # Summary stats
    n_clusters_represented = len(set(selected_df["cluster_id"]))
    print(f"    Clusters represented: {n_clusters_represented}/{len(set(cluster_ids))}")

    return selected_df


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pocket sequence evolution heatmap + diversity selection")
    parser.add_argument("--expansion-root", default=DEFAULT_ROOT)
    parser.add_argument("--initial-csv-dir", default=DEFAULT_INITIAL,
                        help="Directory with initial bile acid CSVs")
    parser.add_argument("--ligands", nargs="+", default=DEFAULT_LIGANDS)
    parser.add_argument("--plot", action="store_true",
                        help="Generate heatmap figures")
    parser.add_argument("--select-diverse", type=int, default=0,
                        help="Number of diverse designs to select per ligand")
    parser.add_argument("--min-hamming", type=int, default=3,
                        help="Min Hamming distance for new cluster (default: 3)")
    parser.add_argument("--fig-dir", default=None,
                        help="Output directory for figures")
    args = parser.parse_args()

    root = Path(args.expansion_root)

    for lig in args.ligands:
        print(f"\n{'='*70}")
        print(f"  {lig.upper()}")
        print(f"{'='*70}")

        # Part 1: Heatmap
        if args.plot:
            round_freqs = compute_per_round_frequencies(lig, root, args.initial_csv_dir)
            if round_freqs:
                fig_dir = Path(args.fig_dir) if args.fig_dir else root / lig
                out_path = fig_dir / f"pocket_evolution_{lig}.png"
                plot_pocket_evolution(lig, round_freqs, out_path)
            else:
                print(f"  {lig.upper()}: no round data found for heatmap")

        # Part 2: Diversity selection
        if args.select_diverse > 0:
            run_diversity_selection(
                lig, root, args.initial_csv_dir,
                args.select_diverse, args.min_hamming
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
