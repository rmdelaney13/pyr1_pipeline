#!/usr/bin/env python
"""
Audit docking completeness across pair cache directories.

Finds pairs where docking was marked complete but not all array tasks
actually finished (race condition from pre-fix orchestrator).

For affected pairs, resets docking/clustering/relax stages in metadata.json
so the orchestrator will re-process them correctly.

Usage:
    # Dry run (report only):
    python audit_docking_completeness.py /scratch/alpine/ryde3462/ml_dataset/tier2_win_ssm_graded

    # Fix affected pairs:
    python audit_docking_completeness.py /scratch/alpine/ryde3462/ml_dataset/tier2_win_ssm_graded --fix
"""

import argparse
import configparser
import json
import sys
from pathlib import Path


def audit_pair(pair_dir: Path) -> dict:
    """Check a single pair for docking completeness."""
    docking_dir = pair_dir / 'docking'
    metadata_path = pair_dir / 'metadata.json'

    result = {
        'pair_id': pair_dir.name,
        'has_docking_dir': docking_dir.exists(),
        'csvs_found': 0,
        'expected_tasks': None,
        'docking_marked_complete': False,
        'clustering_marked_complete': False,
        'relax_marked_complete': False,
        'needs_fix': False,
    }

    if not docking_dir.exists():
        return result

    # Count geometry CSVs
    csvs = list(docking_dir.glob('hbond_geometry_summary*.csv'))
    result['csvs_found'] = len(csvs)

    # Read expected array task count from docking config
    config_path = docking_dir / 'docking_config.txt'
    if config_path.exists():
        try:
            cfg = configparser.ConfigParser()
            cfg.read(str(config_path))
            result['expected_tasks'] = cfg.getint('mutant_docking', 'ArrayTaskCount', fallback=None)
        except Exception:
            pass

    # Check metadata for stage completion
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            stages = meta.get('stages', {})
            result['docking_marked_complete'] = stages.get('docking', {}).get('status') == 'complete'
            result['clustering_marked_complete'] = stages.get('clustering', {}).get('status') == 'complete'
            result['relax_marked_complete'] = stages.get('relax', {}).get('status') == 'complete'
        except Exception:
            pass

    # Determine if this pair needs fixing
    if (result['docking_marked_complete']
            and result['expected_tasks'] is not None
            and result['csvs_found'] < result['expected_tasks']):
        result['needs_fix'] = True

    return result


def fix_pair(pair_dir: Path):
    """Reset docking/clustering/relax stages in metadata.json."""
    metadata_path = pair_dir / 'metadata.json'
    if not metadata_path.exists():
        return False

    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    stages = meta.get('stages', {})
    changed = False

    for stage in ['docking', 'clustering', 'relax']:
        if stage in stages and stages[stage].get('status') == 'complete':
            stages[stage] = {}  # Reset stage
            changed = True

    if changed:
        meta['stages'] = stages
        with open(metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)

    return changed


def main():
    parser = argparse.ArgumentParser(description='Audit docking completeness in ML pipeline cache')
    parser.add_argument('cache_dirs', nargs='+', help='Cache directory(ies) to audit')
    parser.add_argument('--fix', action='store_true', help='Reset affected pairs (default: dry run)')
    args = parser.parse_args()

    total_pairs = 0
    affected_pairs = 0
    fixed_pairs = 0

    for cache_dir_str in args.cache_dirs:
        cache_dir = Path(cache_dir_str)
        if not cache_dir.exists():
            print(f"WARNING: {cache_dir} does not exist, skipping")
            continue

        print(f"\nAuditing: {cache_dir}")
        print("-" * 70)

        # Find all pair directories (may be nested under batch subdirs)
        pair_dirs = sorted(cache_dir.rglob('pair_*/metadata.json'))

        for meta_path in pair_dirs:
            pair_dir = meta_path.parent
            total_pairs += 1
            result = audit_pair(pair_dir)

            if result['needs_fix']:
                affected_pairs += 1
                print(f"  INCOMPLETE: {pair_dir.name} — "
                      f"{result['csvs_found']}/{result['expected_tasks']} CSVs, "
                      f"docking={'done' if result['docking_marked_complete'] else 'pending'}, "
                      f"clustering={'done' if result['clustering_marked_complete'] else 'pending'}, "
                      f"relax={'done' if result['relax_marked_complete'] else 'pending'}")

                if args.fix:
                    if fix_pair(pair_dir):
                        fixed_pairs += 1
                        print(f"           → FIXED (reset docking/clustering/relax stages)")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_pairs} pairs audited, {affected_pairs} with incomplete docking")
    if args.fix:
        print(f"  Fixed: {fixed_pairs} pairs (stages reset, will re-process on next run)")
    else:
        if affected_pairs > 0:
            print(f"  Run with --fix to reset affected pairs")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
