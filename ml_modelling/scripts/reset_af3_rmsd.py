#!/usr/bin/env python
"""
Reset AF3 summaries where RMSD_min == RMSD_bestdG (old single-structure code).

Finds pairs where the old orchestrator wrote the same value for both fields,
deletes their summary.json, and resets AF3 stage metadata so the orchestrator
recalculates with true min RMSD across all relaxed structures.

Usage:
    # Dry run (report only):
    python reset_af3_rmsd.py /scratch/alpine/ryde3462/ml_dataset/tier1_strong_binders

    # Fix affected pairs:
    python reset_af3_rmsd.py /scratch/alpine/ryde3462/ml_dataset/tier1_strong_binders --fix

    # Multiple tiers:
    python reset_af3_rmsd.py /scratch/alpine/ryde3462/ml_dataset/tier1_strong_binders \
                             /scratch/alpine/ryde3462/ml_dataset/tier2_win_ssm_graded --fix
"""

import argparse
import json
import sys
from pathlib import Path


def check_summary(summary_path: Path) -> bool:
    """Return True if this summary needs recalculation (min == bestdG)."""
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)

        rmsd_min = data.get('ligand_RMSD_to_template_min')
        rmsd_dg = data.get('ligand_RMSD_to_template_bestdG')

        # Skip if either is None (incomplete data)
        if rmsd_min is None or rmsd_dg is None:
            return False

        # Flag if they're identical (old single-structure code)
        return rmsd_min == rmsd_dg

    except Exception:
        return False


def reset_af3_stage(metadata_path: Path, mode: str) -> bool:
    """Reset af3_binary or af3_ternary stage in metadata.json."""
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)

        stage_name = f'af3_{mode}'
        stages = meta.get('stages', {})
        if stage_name in stages and stages[stage_name].get('status') == 'complete':
            stages[stage_name] = {}
            meta['stages'] = stages
            with open(metadata_path, 'w') as f:
                json.dump(meta, f, indent=2)
            return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Reset AF3 summaries where RMSD_min == RMSD_bestdG')
    parser.add_argument('cache_dirs', nargs='+', help='Cache directory(ies) to scan')
    parser.add_argument('--fix', action='store_true',
                        help='Delete summaries and reset metadata (default: dry run)')
    args = parser.parse_args()

    total_checked = 0
    needs_reset = 0
    reset_count = 0

    for cache_dir_str in args.cache_dirs:
        cache_dir = Path(cache_dir_str)
        if not cache_dir.exists():
            print(f"WARNING: {cache_dir} does not exist, skipping")
            continue

        print(f"\nScanning: {cache_dir}")
        print("-" * 70)

        for summary_path in sorted(cache_dir.rglob('af3_*/summary.json')):
            total_checked += 1
            pair_dir = summary_path.parent.parent
            mode = summary_path.parent.name.replace('af3_', '')  # binary or ternary

            if not check_summary(summary_path):
                continue

            needs_reset += 1

            # Read values for reporting
            with open(summary_path, 'r') as f:
                data = json.load(f)
            rmsd_val = data.get('ligand_RMSD_to_template_min')

            print(f"  NEEDS RESET: {pair_dir.name}/af3_{mode} — "
                  f"RMSD_min == RMSD_bestdG == {rmsd_val}")

            if args.fix:
                # Delete summary.json
                summary_path.unlink()

                # Reset stage in metadata
                metadata_path = pair_dir / 'metadata.json'
                if reset_af3_stage(metadata_path, mode):
                    reset_count += 1
                    print(f"             → FIXED (summary deleted, stage reset)")
                else:
                    print(f"             → summary deleted, metadata unchanged")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_checked} summaries checked, "
          f"{needs_reset} need recalculation (RMSD_min == RMSD_bestdG)")
    if args.fix:
        print(f"  Reset: {reset_count} stages (will recalculate on next orchestrator run)")
    else:
        if needs_reset > 0:
            print(f"  Run with --fix to delete summaries and reset stages")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
