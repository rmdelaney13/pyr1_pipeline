#!/usr/bin/env python
"""
Aggregate all Rosetta + AF3 outputs into ML feature table.

This script extracts features from:
  - Conformer generation (diversity, energy)
  - Docking (cluster statistics, best score, convergence)
  - Relax (interface energy, buried unsats, H-bonds)
  - AF3 binary (ipTM, pLDDT, interface PAE)
  - AF3 ternary (ipTM, pLDDT, water H-bonds)

Output: features_table.csv with ~45 columns per (ligand, variant) pair

Author: Claude Code (Whitehead Lab PYR1 Pipeline)
Date: 2026-02-16
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def extract_conformer_features(pair_cache: Path) -> Dict:
    """
    Extract ligand conformer generation features.

    Expected files:
        pair_cache/conformers/conformer_report.csv
        pair_cache/conformers/metadata.json

    Returns dict with:
        - conformer_count: Number of final conformers
        - conformer_min_energy: Lowest MMFF energy
        - conformer_max_rmsd: Maximum pairwise RMSD (diversity)
    """
    conformer_dir = pair_cache / 'conformers'
    report_csv = conformer_dir / 'conformer_report.csv'
    metadata_json = conformer_dir / 'metadata.json'

    if not report_csv.exists():
        return {
            'conformer_status': 'MISSING',
            'conformer_count': np.nan,
            'conformer_min_energy': np.nan,
            'conformer_max_rmsd': np.nan,
        }

    try:
        df = pd.read_csv(report_csv)

        return {
            'conformer_status': 'COMPLETE',
            'conformer_count': len(df),
            'conformer_min_energy': df['mmff_energy'].min() if 'mmff_energy' in df.columns else np.nan,
            'conformer_max_rmsd': df['rmsd_to_centroid'].max() if 'rmsd_to_centroid' in df.columns else np.nan,
        }
    except Exception as e:
        logger.warning(f"Failed to parse conformer report: {e}")
        return {
            'conformer_status': 'ERROR',
            'conformer_count': np.nan,
            'conformer_min_energy': np.nan,
            'conformer_max_rmsd': np.nan,
        }


def extract_docking_features(pair_cache: Path) -> Dict:
    """
    Extract docking cluster statistics (ML-critical features!).

    Expected files:
        pair_cache/docking/clustering_stats.json
        pair_cache/docking/clustering_stats.csv

    Returns dict with:
        - docking_best_score: Best Rosetta score across all poses
        - docking_cluster_1_size: Size of largest cluster
        - docking_convergence_ratio: Fraction in largest cluster (KEY METRIC)
        - docking_num_clusters: Total number of clusters
        - docking_score_range: Energy landscape breadth
        - docking_clash_flag: 1 if best_score > 0 (clash)
        - docking_cluster_1_rmsd: Intra-cluster RMSD (tightness)
    """
    docking_dir = pair_cache / 'docking'
    stats_json = docking_dir / 'clustering_stats.json'

    if not stats_json.exists():
        return {
            'docking_status': 'MISSING',
            'docking_best_score': np.nan,
            'docking_cluster_1_size': np.nan,
            'docking_convergence_ratio': np.nan,
            'docking_num_clusters': np.nan,
            'docking_score_range': np.nan,
            'docking_clash_flag': np.nan,
            'docking_cluster_1_rmsd': np.nan,
        }

    try:
        with open(stats_json, 'r') as f:
            data = json.load(f)

        global_stats = data.get('global_stats', {})
        clusters = data.get('clusters', [])

        # Extract global stats
        best_score = global_stats.get('best_overall_score', np.nan)
        convergence_ratio = global_stats.get('convergence_ratio', np.nan)
        num_clusters = global_stats.get('num_clusters', np.nan)
        score_range = global_stats.get('score_range', np.nan)
        clash_flag = 1 if best_score > 0 else 0

        # Extract top cluster stats
        if clusters:
            top_cluster = clusters[0]
            cluster_1_size = top_cluster.get('size', np.nan)
            cluster_1_rmsd = top_cluster.get('intra_cluster_rmsd', np.nan)
        else:
            cluster_1_size = np.nan
            cluster_1_rmsd = np.nan

        return {
            'docking_status': 'COMPLETE',
            'docking_best_score': best_score,
            'docking_cluster_1_size': cluster_1_size,
            'docking_convergence_ratio': convergence_ratio,
            'docking_num_clusters': num_clusters,
            'docking_score_range': score_range,
            'docking_clash_flag': clash_flag,
            'docking_cluster_1_rmsd': cluster_1_rmsd,
        }

    except Exception as e:
        logger.warning(f"Failed to parse docking stats: {e}")
        return {
            'docking_status': 'ERROR',
            'docking_best_score': np.nan,
            'docking_cluster_1_size': np.nan,
            'docking_convergence_ratio': np.nan,
            'docking_num_clusters': np.nan,
            'docking_score_range': np.nan,
            'docking_clash_flag': np.nan,
            'docking_cluster_1_rmsd': np.nan,
        }


def extract_rosetta_relax_features(pair_cache: Path) -> Dict:
    """
    Extract Rosetta relax features from score file.

    Expected files:
        pair_cache/relax/rosetta_scores.sc

    Returns dict with:
        - rosetta_total_score: Total Rosetta energy
        - rosetta_dG_sep: Interface binding energy (ΔΔG)
        - rosetta_buried_unsats: Buried unsatisfied H-bonds
        - rosetta_sasa_interface: Interface SASA
        - rosetta_hbonds_interface: H-bonds across interface
        - rosetta_ligand_neighbors: Residues within 4Å of ligand
    """
    relax_dir = pair_cache / 'relax'
    score_file = relax_dir / 'rosetta_scores.sc'

    if not score_file.exists():
        return {
            'rosetta_status': 'MISSING',
            'rosetta_total_score': np.nan,
            'rosetta_dG_sep': np.nan,
            'rosetta_buried_unsats': np.nan,
            'rosetta_sasa_interface': np.nan,
            'rosetta_hbonds_interface': np.nan,
            'rosetta_ligand_neighbors': np.nan,
        }

    try:
        # Parse Rosetta score file (space-delimited)
        df = pd.read_csv(score_file, delim_whitespace=True, comment='#', skiprows=1)

        # Extract key metrics (take best if multiple lines)
        features = {
            'rosetta_status': 'COMPLETE',
            'rosetta_total_score': df['total_score'].min() if 'total_score' in df.columns else np.nan,
            'rosetta_dG_sep': df['dG_separated'].min() if 'dG_separated' in df.columns else np.nan,
            'rosetta_buried_unsats': df['buried_unsatisfied_hbonds'].min() if 'buried_unsatisfied_hbonds' in df.columns else np.nan,
            'rosetta_sasa_interface': df['interface_sasa'].max() if 'interface_sasa' in df.columns else np.nan,
            'rosetta_hbonds_interface': df['interface_hbonds'].max() if 'interface_hbonds' in df.columns else np.nan,
            'rosetta_ligand_neighbors': df['ligand_neighbors'].max() if 'ligand_neighbors' in df.columns else np.nan,
        }

        return features

    except Exception as e:
        logger.warning(f"Failed to parse Rosetta scores: {e}")
        return {
            'rosetta_status': 'ERROR',
            'rosetta_total_score': np.nan,
            'rosetta_dG_sep': np.nan,
            'rosetta_buried_unsats': np.nan,
            'rosetta_sasa_interface': np.nan,
            'rosetta_hbonds_interface': np.nan,
            'rosetta_ligand_neighbors': np.nan,
        }


def extract_af3_features(pair_cache: Path, mode: str = 'binary') -> Dict:
    """
    Extract AlphaFold3 prediction features.

    Expected files:
        pair_cache/af3_binary/summary.json
        pair_cache/af3_ternary/summary.json

    Args:
        pair_cache: Path to pair cache directory
        mode: 'binary' or 'ternary'

    Returns dict with:
        - af3_{mode}_ipTM: Interface predicted TM-score
        - af3_{mode}_pLDDT_protein: Mean protein pLDDT
        - af3_{mode}_pLDDT_ligand: Mean ligand pLDDT
        - af3_{mode}_interface_PAE: Mean interface PAE
        - af3_{mode}_ligand_RMSD: Ligand RMSD to template
    """
    af3_dir = pair_cache / f'af3_{mode}'
    summary_json = af3_dir / 'summary.json'

    prefix = f'af3_{mode}_'

    if not summary_json.exists():
        return {
            f'{prefix}status': 'MISSING',
            f'{prefix}ipTM': np.nan,
            f'{prefix}pLDDT_protein': np.nan,
            f'{prefix}pLDDT_ligand': np.nan,
            f'{prefix}interface_PAE': np.nan,
            f'{prefix}ligand_RMSD': np.nan,
        }

    try:
        with open(summary_json, 'r') as f:
            data = json.load(f)

        return {
            f'{prefix}status': 'COMPLETE',
            f'{prefix}ipTM': data.get('ipTM', np.nan),
            f'{prefix}pLDDT_protein': data.get('mean_pLDDT_protein', np.nan),
            f'{prefix}pLDDT_ligand': data.get('mean_pLDDT_ligand', np.nan),
            f'{prefix}interface_PAE': data.get('mean_interface_PAE', np.nan),
            f'{prefix}ligand_RMSD': data.get('ligand_RMSD_to_template', np.nan),
        }

    except Exception as e:
        logger.warning(f"Failed to parse AF3 {mode} summary: {e}")
        return {
            f'{prefix}status': 'ERROR',
            f'{prefix}ipTM': np.nan,
            f'{prefix}pLDDT_protein': np.nan,
            f'{prefix}pLDDT_ligand': np.nan,
            f'{prefix}interface_PAE': np.nan,
            f'{prefix}ligand_RMSD': np.nan,
        }


def extract_all_features(pair_cache: Path, pair_metadata: Dict) -> Dict:
    """
    Extract all features for a single (ligand, variant) pair.

    Args:
        pair_cache: Path to pair cache directory
        pair_metadata: Dict with pair_id, ligand_name, variant_name, etc.

    Returns:
        Dict with ~45 features
    """
    features = {}

    # Add input metadata
    features.update({
        'pair_id': pair_metadata.get('pair_id', ''),
        'ligand_name': pair_metadata.get('ligand_name', ''),
        'ligand_smiles': pair_metadata.get('ligand_smiles', ''),
        'variant_name': pair_metadata.get('variant_name', ''),
        'variant_signature': pair_metadata.get('variant_signature', ''),
        'label': pair_metadata.get('label', np.nan),
        'label_tier': pair_metadata.get('label_tier', ''),
        'label_source': pair_metadata.get('label_source', ''),
        'label_confidence': pair_metadata.get('label_confidence', ''),
        'affinity_EC50_uM': pair_metadata.get('affinity_EC50_uM', np.nan),
    })

    # Extract conformer features
    features.update(extract_conformer_features(pair_cache))

    # Extract docking cluster statistics (CRITICAL for ML!)
    features.update(extract_docking_features(pair_cache))

    # Extract Rosetta relax features
    features.update(extract_rosetta_relax_features(pair_cache))

    # Extract AF3 binary features
    features.update(extract_af3_features(pair_cache, mode='binary'))

    # Extract AF3 ternary features
    features.update(extract_af3_features(pair_cache, mode='ternary'))

    return features


# ═══════════════════════════════════════════════════════════════════
# MAIN AGGREGATION
# ═══════════════════════════════════════════════════════════════════

def aggregate_features(
    cache_dir: Path,
    pairs_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """
    Aggregate features for all pairs in dataset.

    Args:
        cache_dir: Base cache directory containing pair subdirectories
        pairs_csv: CSV with pair metadata (pair_id, ligand_name, variant_name, label, etc.)
        output_csv: Output features CSV path

    Returns:
        DataFrame with features for all pairs
    """
    logger.info(f"Loading pairs metadata from {pairs_csv}...")
    pairs_df = pd.read_csv(pairs_csv)
    logger.info(f"Loaded {len(pairs_df)} pairs")

    all_features = []

    for idx, row in pairs_df.iterrows():
        pair_id = row['pair_id']
        pair_cache = cache_dir / pair_id

        logger.info(f"[{idx+1}/{len(pairs_df)}] Extracting features for {pair_id}...")

        # Extract features
        features = extract_all_features(pair_cache, row.to_dict())
        all_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    # Save output
    features_df.to_csv(output_csv, index=False)
    logger.info(f"✓ Saved {len(features_df)} feature rows to {output_csv}")

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("=" * 60)

    for stage in ['conformer', 'docking', 'rosetta', 'af3_binary', 'af3_ternary']:
        status_col = f'{stage}_status'
        if status_col in features_df.columns:
            counts = features_df[status_col].value_counts()
            logger.info(f"{stage.upper()}: {counts.to_dict()}")

    # Completeness analysis
    logger.info("\nCOMPLETENESS:")
    for col in features_df.columns:
        if col.endswith('_status') or col in ['pair_id', 'ligand_name', 'variant_name']:
            continue
        missing_pct = (features_df[col].isna().sum() / len(features_df)) * 100
        if missing_pct > 0:
            logger.info(f"  {col}: {100-missing_pct:.1f}% complete")

    logger.info("=" * 60)

    return features_df


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate ML features from pipeline outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--cache-dir', required=True, help='Base cache directory')
    parser.add_argument('--pairs-csv', required=True, help='Pairs metadata CSV (with pair_id, label, etc.)')
    parser.add_argument('--output', required=True, help='Output features CSV')

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        sys.exit(1)

    if not os.path.exists(args.pairs_csv):
        logger.error(f"Pairs CSV not found: {args.pairs_csv}")
        sys.exit(1)

    # Aggregate features
    features_df = aggregate_features(
        cache_dir=cache_dir,
        pairs_csv=args.pairs_csv,
        output_csv=args.output
    )

    logger.info(f"\n✓ Feature aggregation complete! Output: {args.output}")


if __name__ == '__main__':
    main()
