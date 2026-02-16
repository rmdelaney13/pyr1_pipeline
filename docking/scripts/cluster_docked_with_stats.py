#!/usr/bin/env python
"""
Post-array clustering with DETAILED STATISTICS for ML dataset generation.

This script aggregates docking outputs from SLURM array tasks and performs
global clustering, but ALSO outputs cluster statistics critical for ML:
  - Convergence ratio (fraction in largest cluster)
  - Number of clusters (diversity metric)
  - Intra-cluster RMSD (tightness)
  - Score range (energy landscape breadth)
  - Clash detection (positive Rosetta scores)

These statistics help distinguish:
  - Strong binders: High convergence, low diversity, narrow funnel
  - Weak/non-binders: Low convergence, high diversity, scattered poses
  - Clashes: Positive scores, low convergence

Usage:
  python cluster_docked_with_stats.py \
      --input-dir /path/to/docking/output/ \
      --output-dir /path/to/clustered/ \
      --rmsd-cutoff 2.0 \
      --stats-csv clustering_stats.csv

Output files:
  - best_pose.pdb: Lowest-scoring pose from largest cluster
  - clustering_stats.csv: Detailed statistics for ML features
  - cluster_<N>.pdb: Representative pose from each cluster (optional)

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
Date: 2026-02-16
"""

import argparse
import csv
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import json

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PDB PARSING
# ═══════════════════════════════════════════════════════════════════

def _is_heavy_atom(line):
    """Check if PDB ATOM/HETATM line is a heavy atom (not hydrogen)."""
    elem = line[76:78].strip()
    if not elem:
        name = line[12:16].strip()
        elem = "".join([c for c in name if c.isalpha()])[:1]
    return elem.upper() != "H"


def _parse_residue_key(line):
    """Extract residue key from PDB line."""
    chain = line[21:22]
    resseq = line[22:26]
    icode = line[26:27]
    resname = line[17:20]
    return (chain, resseq, icode, resname)


def extract_ligand_coords(pdb_path: str, ligand_resname: str = None) -> np.ndarray:
    """
    Extract ligand heavy atom coordinates from PDB file.

    Args:
        pdb_path: Path to PDB file
        ligand_resname: Ligand residue name (default: auto-detect largest HETATM)

    Returns:
        np.array of shape (N, 3) with ligand heavy atom coords
    """
    residue_heavy_coords = {}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue

            resname = line[17:20].strip().upper()

            # Skip water
            if resname in {"HOH", "WAT", "TP3"}:
                continue

            # Filter by ligand resname if specified
            if ligand_resname and resname != ligand_resname.upper():
                continue

            # Only heavy atoms
            if not _is_heavy_atom(line):
                continue

            key = _parse_residue_key(line)

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            residue_heavy_coords.setdefault(key, []).append([x, y, z])

    if not residue_heavy_coords:
        return np.zeros((0, 3), dtype=float)

    # Return largest residue (most atoms)
    best_key = max(residue_heavy_coords.keys(), key=lambda k: len(residue_heavy_coords[k]))
    return np.asarray(residue_heavy_coords[best_key], dtype=float)


def calculate_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Calculate RMSD between two sets of coordinates.

    Args:
        coords_a, coords_b: np.arrays of shape (N, 3)

    Returns:
        RMSD value (Angstroms)
    """
    if coords_a.shape != coords_b.shape or coords_a.size == 0:
        return float("inf")

    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# ═══════════════════════════════════════════════════════════════════
# SCORE COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_docking_results(input_dir: str, pattern_prefix: str = "docking_results_task_") -> pd.DataFrame:
    """
    Collect all docking results CSVs from array tasks.

    Args:
        input_dir: Directory containing docking_results_task_*.csv files
        pattern_prefix: Prefix for CSV files

    Returns:
        DataFrame with columns: conformer_id, repeat, score, output_pdb
    """
    all_results = []

    for filename in os.listdir(input_dir):
        if not filename.startswith(pattern_prefix) or not filename.endswith(".csv"):
            continue

        csv_path = os.path.join(input_dir, filename)
        logger.info(f"Reading {filename}...")

        try:
            df = pd.read_csv(csv_path)
            all_results.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {filename}: {e}")
            continue

    if not all_results:
        raise FileNotFoundError(f"No CSV files found matching {pattern_prefix}*.csv in {input_dir}")

    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"Collected {len(combined_df)} total poses from {len(all_results)} CSV files")

    return combined_df


# ═══════════════════════════════════════════════════════════════════
# CLUSTERING WITH STATISTICS
# ═══════════════════════════════════════════════════════════════════

def cluster_poses_by_rmsd(
    results_df: pd.DataFrame,
    rmsd_cutoff: float = 2.0,
    ligand_resname: str = None
) -> List[Dict]:
    """
    Cluster docked poses by ligand RMSD.

    Args:
        results_df: DataFrame with 'output_pdb' and 'score' columns
        rmsd_cutoff: RMSD threshold for clustering (Angstroms)
        ligand_resname: Ligand residue name (default: auto-detect)

    Returns:
        List of cluster dicts with:
            - cluster_id: int
            - members: List of row indices
            - centroid_idx: Index of centroid pose
            - best_score: Lowest score in cluster
            - mean_score: Average score
            - intra_cluster_rmsd: Average RMSD within cluster
    """
    # Extract ligand coordinates for all poses
    logger.info(f"Extracting ligand coordinates from {len(results_df)} poses...")

    coords_list = []
    valid_indices = []

    for idx, row in results_df.iterrows():
        pdb_path = row['output_pdb']

        if not os.path.exists(pdb_path):
            logger.warning(f"PDB not found: {pdb_path}")
            continue

        coords = extract_ligand_coords(pdb_path, ligand_resname)

        if coords.size == 0:
            logger.warning(f"No ligand coords in {pdb_path}")
            continue

        coords_list.append(coords)
        valid_indices.append(idx)

    if not coords_list:
        raise ValueError("No valid ligand coordinates found in any PDB!")

    logger.info(f"Extracted coords from {len(coords_list)} poses")

    # Greedy clustering by RMSD
    logger.info(f"Clustering with RMSD cutoff {rmsd_cutoff} Å...")

    clusters = []
    clustered = set()

    for i, idx_i in enumerate(valid_indices):
        if idx_i in clustered:
            continue

        # Start new cluster
        cluster_members = [idx_i]
        clustered.add(idx_i)

        # Find all poses within RMSD cutoff
        for j, idx_j in enumerate(valid_indices):
            if idx_j in clustered:
                continue

            rmsd = calculate_rmsd(coords_list[i], coords_list[j])

            if rmsd <= rmsd_cutoff:
                cluster_members.append(idx_j)
                clustered.add(idx_j)

        # Compute cluster statistics
        cluster_df = results_df.loc[cluster_members]

        # Find centroid (pose with minimum average RMSD to others)
        if len(cluster_members) > 1:
            avg_rmsds = []
            for idx_k in cluster_members:
                k_idx_in_coords = valid_indices.index(idx_k)
                rmsds_to_others = [
                    calculate_rmsd(coords_list[k_idx_in_coords], coords_list[valid_indices.index(idx_m)])
                    for idx_m in cluster_members if idx_m != idx_k
                ]
                avg_rmsds.append(np.mean(rmsds_to_others))

            centroid_idx = cluster_members[np.argmin(avg_rmsds)]
            intra_rmsd = np.mean(avg_rmsds)
        else:
            centroid_idx = cluster_members[0]
            intra_rmsd = 0.0

        clusters.append({
            'cluster_id': len(clusters),
            'members': cluster_members,
            'size': len(cluster_members),
            'centroid_idx': centroid_idx,
            'best_score': cluster_df['score'].min(),
            'mean_score': cluster_df['score'].mean(),
            'intra_cluster_rmsd': intra_rmsd,
        })

    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: c['size'], reverse=True)

    # Re-assign cluster IDs after sorting
    for i, cluster in enumerate(clusters):
        cluster['cluster_id'] = i

    logger.info(f"Found {len(clusters)} clusters")
    logger.info(f"Largest cluster: {clusters[0]['size']} poses ({clusters[0]['size']/len(results_df)*100:.1f}%)")

    return clusters


def compute_clustering_statistics(clusters: List[Dict], total_poses: int) -> Dict:
    """
    Compute global statistics from clustering results.

    Returns dict with ML-relevant features:
        - total_poses: Total number of poses
        - num_clusters: Number of clusters
        - largest_cluster_size: Size of largest cluster
        - convergence_ratio: Fraction of poses in largest cluster
        - top3_cluster_sizes: Sizes of top 3 clusters
        - best_overall_score: Lowest score across all clusters
        - score_range: Difference between best and worst scores
        - clash_count: Number of poses with positive scores (clash)
        - clash_fraction: Fraction of poses with positive scores
    """
    if not clusters:
        return {
            'total_poses': total_poses,
            'num_clusters': 0,
            'largest_cluster_size': 0,
            'convergence_ratio': 0.0,
            'top3_cluster_sizes': [0, 0, 0],
            'best_overall_score': np.nan,
            'score_range': np.nan,
            'clash_count': 0,
            'clash_fraction': 0.0,
        }

    # Collect all scores
    all_scores = [c['best_score'] for c in clusters]
    best_overall = min(all_scores)
    worst_overall = max(all_scores)

    # Top-3 cluster sizes
    top3_sizes = [c['size'] for c in clusters[:3]]
    while len(top3_sizes) < 3:
        top3_sizes.append(0)

    # Clash detection (positive scores = steric clash)
    clash_count = sum(1 for c in clusters if c['best_score'] > 0)
    clash_fraction = clash_count / len(clusters) if clusters else 0.0

    stats = {
        'total_poses': total_poses,
        'num_clusters': len(clusters),
        'largest_cluster_size': clusters[0]['size'],
        'convergence_ratio': clusters[0]['size'] / total_poses,
        'top3_cluster_sizes': top3_sizes,
        'best_overall_score': best_overall,
        'score_range': worst_overall - best_overall,
        'clash_count': clash_count,
        'clash_fraction': clash_fraction,
    }

    return stats


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════

def save_clustering_results(
    clusters: List[Dict],
    results_df: pd.DataFrame,
    output_dir: str,
    stats_csv: str = "clustering_stats.csv",
    save_all_clusters: bool = False
) -> None:
    """
    Save clustering results and statistics.

    Outputs:
        - best_pose.pdb: Centroid of largest cluster
        - clustering_stats.csv: Per-cluster and global statistics
        - cluster_<N>.pdb: Centroids of all clusters (if save_all_clusters=True)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save best pose (centroid of largest cluster)
    best_cluster = clusters[0]
    best_pdb_src = results_df.loc[best_cluster['centroid_idx'], 'output_pdb']
    best_pdb_dst = os.path.join(output_dir, 'best_pose.pdb')

    import shutil
    shutil.copy(best_pdb_src, best_pdb_dst)
    logger.info(f"Saved best pose: {best_pdb_dst}")

    # Save cluster centroids (optional)
    if save_all_clusters:
        for cluster in clusters:
            cluster_pdb_src = results_df.loc[cluster['centroid_idx'], 'output_pdb']
            cluster_pdb_dst = os.path.join(output_dir, f"cluster_{cluster['cluster_id']}.pdb")
            shutil.copy(cluster_pdb_src, cluster_pdb_dst)

    # Compute global statistics
    global_stats = compute_clustering_statistics(clusters, len(results_df))

    # Save statistics CSV
    stats_path = os.path.join(output_dir, stats_csv)

    with open(stats_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'cluster_id', 'cluster_size', 'best_score', 'mean_score',
            'intra_cluster_rmsd', 'centroid_pdb'
        ])

        # Per-cluster rows
        for cluster in clusters:
            centroid_pdb = results_df.loc[cluster['centroid_idx'], 'output_pdb']
            writer.writerow([
                cluster['cluster_id'],
                cluster['size'],
                f"{cluster['best_score']:.3f}",
                f"{cluster['mean_score']:.3f}",
                f"{cluster['intra_cluster_rmsd']:.3f}",
                centroid_pdb
            ])

        # Global statistics row
        writer.writerow([])  # Blank line
        writer.writerow(['GLOBAL_STATISTICS', '', '', '', '', ''])
        writer.writerow(['total_poses', global_stats['total_poses'], '', '', '', ''])
        writer.writerow(['num_clusters', global_stats['num_clusters'], '', '', '', ''])
        writer.writerow(['largest_cluster_size', global_stats['largest_cluster_size'], '', '', '', ''])
        writer.writerow(['convergence_ratio', f"{global_stats['convergence_ratio']:.4f}", '', '', '', ''])
        writer.writerow(['top3_cluster_sizes', str(global_stats['top3_cluster_sizes']), '', '', '', ''])
        writer.writerow(['best_overall_score', f"{global_stats['best_overall_score']:.3f}", '', '', '', ''])
        writer.writerow(['score_range', f"{global_stats['score_range']:.3f}", '', '', '', ''])
        writer.writerow(['clash_count', global_stats['clash_count'], '', '', '', ''])
        writer.writerow(['clash_fraction', f"{global_stats['clash_fraction']:.4f}", '', '', '', ''])

    logger.info(f"Saved clustering statistics: {stats_path}")

    # Also save as JSON for easy parsing
    json_path = os.path.join(output_dir, 'clustering_stats.json')
    with open(json_path, 'w') as f:
        json.dump({
            'clusters': [
                {
                    'cluster_id': c['cluster_id'],
                    'size': c['size'],
                    'best_score': c['best_score'],
                    'mean_score': c['mean_score'],
                    'intra_cluster_rmsd': c['intra_cluster_rmsd'],
                }
                for c in clusters
            ],
            'global_stats': global_stats
        }, f, indent=2)

    logger.info(f"Saved JSON statistics: {json_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Cluster docked poses with detailed statistics for ML',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input-dir', required=True, help='Directory with docking results CSVs')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--rmsd-cutoff', type=float, default=2.0, help='RMSD clustering cutoff (Å)')
    parser.add_argument('--stats-csv', default='clustering_stats.csv', help='Output statistics CSV filename')
    parser.add_argument('--ligand-resname', help='Ligand residue name (default: auto-detect)')
    parser.add_argument('--save-all-clusters', action='store_true', help='Save PDBs for all clusters')
    parser.add_argument('--pattern-prefix', default='docking_results_task_', help='CSV filename prefix')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CLUSTERING WITH STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Input dir: {args.input_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"RMSD cutoff: {args.rmsd_cutoff} Å")
    logger.info("=" * 60)

    # Collect results
    results_df = collect_docking_results(args.input_dir, args.pattern_prefix)

    # Cluster poses
    clusters = cluster_poses_by_rmsd(
        results_df,
        rmsd_cutoff=args.rmsd_cutoff,
        ligand_resname=args.ligand_resname
    )

    # Save results
    save_clustering_results(
        clusters,
        results_df,
        args.output_dir,
        stats_csv=args.stats_csv,
        save_all_clusters=args.save_all_clusters
    )

    # Print summary
    global_stats = compute_clustering_statistics(clusters, len(results_df))

    logger.info("=" * 60)
    logger.info("CLUSTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total poses: {global_stats['total_poses']}")
    logger.info(f"Number of clusters: {global_stats['num_clusters']}")
    logger.info(f"Largest cluster: {global_stats['largest_cluster_size']} poses")
    logger.info(f"Convergence ratio: {global_stats['convergence_ratio']:.2%}")
    logger.info(f"Best score: {global_stats['best_overall_score']:.2f}")
    logger.info(f"Score range: {global_stats['score_range']:.2f}")

    if global_stats['clash_count'] > 0:
        logger.warning(f"⚠ WARNING: {global_stats['clash_count']} clusters with positive scores (CLASHES!)")
        logger.warning(f"  Clash fraction: {global_stats['clash_fraction']:.2%}")

    logger.info("=" * 60)
    logger.info("✓ Clustering complete!")


if __name__ == '__main__':
    main()
