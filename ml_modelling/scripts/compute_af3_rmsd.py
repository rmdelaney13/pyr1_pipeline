#!/usr/bin/env python3
"""
Compute AF3 ligand RMSD to Rosetta-relaxed structures for a batch of pairs.

Designed to run as a SLURM array task. Each task processes a slice of pairs
from a manifest file.

Usage (standalone):
    python compute_af3_rmsd.py --manifest /path/to/rmsd_manifest.tsv --task-index 0

Usage (SLURM array):
    sbatch --array=0-N submit_af3_rmsd.sh /path/to/rmsd_manifest.tsv

Manifest format (TSV, one row per pair+mode):
    pair_dir\taf3_cif_path\tmode

Outputs:
    Writes summary.json into pair_dir/af3_{mode}/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent paths so we can import from prepare_af3_ml
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def process_single_entry(pair_dir: Path, af3_cif_path: Path, mode: str,
                         ref_model_path: str = None, recompute: bool = False):
    """
    Compute AF3 metrics + ligand RMSD for one pair+mode and write summary.json.
    """
    from prepare_af3_ml import (
        extract_af3_metrics,
        compute_ligand_rmsd_to_rosetta,
        compute_all_ligand_rmsds_to_rosetta,
        compute_hbond_water_geometry,
        find_all_relaxed_pdbs,
        find_best_relaxed_pdb,
        write_summary_json,
    )

    summary_file = pair_dir / f'af3_{mode}' / 'summary.json'

    # In recompute mode, load existing summary and only add H-bond geometry
    if recompute and summary_file.exists():
        with open(summary_file) as f:
            existing = json.load(f)

        # Skip if already has both fields
        has_dist = existing.get('min_dist_to_ligand_O') is not None
        has_angle = existing.get('hbond_water_angle') is not None
        if has_dist and has_angle:
            logger.info(f"  SKIP {pair_dir.name}/{mode} — H-bond geometry already computed")
            return True

        # Compute H-bond water geometry
        if ref_model_path and af3_cif_path.exists():
            geom = compute_hbond_water_geometry(
                af3_cif_path=str(af3_cif_path),
                ref_model_path=ref_model_path,
            )
            existing['min_dist_to_ligand_O'] = geom['distance']
            existing['hbond_water_angle'] = geom['angle']
        else:
            existing.setdefault('min_dist_to_ligand_O', None)
            existing.setdefault('hbond_water_angle', None)

        with open(summary_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"  UPDATED {pair_dir.name}/{mode}: "
                     f"dist={existing.get('min_dist_to_ligand_O')}, "
                     f"angle={existing.get('hbond_water_angle')}")
        return True

    # Normal mode: skip if summary exists
    if summary_file.exists() and not recompute:
        logger.info(f"  SKIP {pair_dir.name}/{mode} — summary.json exists")
        return True

    pair_id = pair_dir.name

    # Extract AF3 confidence metrics
    af3_output_dir = af3_cif_path.parent
    metrics = extract_af3_metrics(
        af3_output_dir=str(af3_output_dir),
        pair_id=pair_id,
        mode=mode,
    )
    if metrics is None:
        logger.warning(f"  FAIL {pair_id}/{mode} — could not extract metrics")
        return False

    # Compute ligand RMSD to all relaxed structures
    ligand_rmsds = {'min': None, 'best_dG': None}
    relaxed_pdbs = find_all_relaxed_pdbs(str(pair_dir))
    best_dg_pdb = find_best_relaxed_pdb(str(pair_dir))

    if relaxed_pdbs and af3_cif_path.exists():
        ligand_rmsds = compute_all_ligand_rmsds_to_rosetta(
            af3_cif_path=str(af3_cif_path),
            relaxed_pdbs=relaxed_pdbs,
            best_dg_pdb=best_dg_pdb,
        )
    elif best_dg_pdb and af3_cif_path.exists():
        rmsd = compute_ligand_rmsd_to_rosetta(
            af3_cif_path=str(af3_cif_path),
            rosetta_pdb_path=str(best_dg_pdb),
        )
        ligand_rmsds = {'min': rmsd, 'best_dG': rmsd}

    # Compute H-bond water geometry (distance + angle)
    hbond_dist = None
    hbond_angle = None
    if ref_model_path and af3_cif_path.exists():
        geom = compute_hbond_water_geometry(
            af3_cif_path=str(af3_cif_path),
            ref_model_path=ref_model_path,
        )
        hbond_dist = geom['distance']
        hbond_angle = geom['angle']

    # Write summary (binary-ternary RMSD is computed later by orchestrator
    # since it needs both modes)
    write_summary_json(str(pair_dir / f'af3_{mode}'), metrics, ligand_rmsds,
                       None, hbond_dist, hbond_angle)

    logger.info(f"  OK {pair_id}/{mode}: ipTM={metrics.get('ipTM')}, "
                f"RMSD_min={ligand_rmsds.get('min')}, "
                f"RMSD_bestdG={ligand_rmsds.get('best_dG')}, "
                f"hbond_dist={hbond_dist}, hbond_angle={hbond_angle}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Compute AF3 ligand RMSD for pairs listed in a manifest')
    parser.add_argument('--manifest', required=True,
                        help='Path to TSV manifest (pair_dir, af3_cif_path, mode)')
    parser.add_argument('--task-index', type=int, default=None,
                        help='SLURM_ARRAY_TASK_ID (0-based). If not given, uses env var.')
    parser.add_argument('--pairs-per-task', type=int, default=1,
                        help='Number of pairs to process per array task')
    parser.add_argument('--ref-model', type=str, default=None,
                        help='Reference PDB for H-bond water geometry (e.g., 3QN1_H2O.pdb)')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute: update existing summary.json with new fields only')
    args = parser.parse_args()

    # Determine task index
    import os
    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # Read manifest
    manifest = Path(args.manifest)
    if not manifest.exists():
        logger.error(f"Manifest not found: {manifest}")
        sys.exit(1)

    entries = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                logger.warning(f"Skipping malformed line: {line}")
                continue
            entries.append((Path(parts[0]), Path(parts[1]), parts[2]))

    # Slice for this task
    start = task_index * args.pairs_per_task
    end = start + args.pairs_per_task
    my_entries = entries[start:end]

    if not my_entries:
        logger.info(f"Task {task_index}: no entries to process (start={start}, total={len(entries)})")
        return

    logger.info(f"Task {task_index}: processing {len(my_entries)} entries "
                f"(indices {start}-{end-1} of {len(entries)})")
    if args.recompute:
        logger.info("  RECOMPUTE mode: updating existing summaries with H-bond geometry")

    n_ok = 0
    n_fail = 0
    for pair_dir, cif_path, mode in my_entries:
        try:
            if process_single_entry(pair_dir, cif_path, mode,
                                    ref_model_path=args.ref_model,
                                    recompute=args.recompute):
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            logger.error(f"  ERROR {pair_dir.name}/{mode}: {e}")
            n_fail += 1

    logger.info(f"Task {task_index} done: {n_ok} OK, {n_fail} failed")


if __name__ == '__main__':
    main()
