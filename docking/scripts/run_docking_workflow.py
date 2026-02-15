#!/usr/bin/env python
"""
Master workflow orchestrator for SDF-to-clustered-docks pipeline.

This script coordinates:
1. create_table.py - Generate params/PDBs and alignment table from SDF
2. grade_conformers_glycine_shaved_docking_multiple_slurm.py - Run docking (per array task)
3. cluster_docked_post_array.py - Cluster final results across all arrays

Usage:
    # Submit via SLURM and wait for completion (chainable with design pipeline)
    python run_docking_workflow.py config.txt --slurm --wait

    # Chain full pipeline: docking → design → AF3
    python run_docking_workflow.py config.txt --slurm --wait && \\
    python design/scripts/run_design_pipeline.py config.txt --wait

    # Submit via SLURM without waiting
    python run_docking_workflow.py config.txt --slurm

    # Single run (no SLURM arrays)
    python run_docking_workflow.py config.txt

    # Simulate array tasks locally (useful for testing)
    python run_docking_workflow.py config.txt --local-arrays 4

    # Run specific array index (called by SLURM job)
    python run_docking_workflow.py config.txt --array-index 0 --skip-clustering

    # Skip table creation (if already done)
    python run_docking_workflow.py config.txt --skip-create-table
"""

import argparse
import os
import subprocess
import sys
import time
from configparser import ConfigParser

try:
    import docking_pipeline_utils as dpu
except ImportError:
    # If dpu not in path, add the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import docking_pipeline_utils as dpu


def _resolve_mode(cli_mode, config):
    """Determine which docking mode to use."""
    if cli_mode != "auto":
        return cli_mode
    mode = dpu.cfg_get(config, "pipeline", "DockingMode", None)
    if mode is None:
        mode = dpu.cfg_get(config, "DEFAULT", "DockingMode", "glycine")
    mode = str(mode).strip().lower()
    aliases = {
        "glycine_shaved": "glycine",
        "glycine-shaved": "glycine",
        "glycine": "glycine",
        "sequence": "sequence",
        "docked_to_sequence": "sequence",
    }
    return aliases.get(mode, mode)


def run_create_table(config_file):
    """Step 1: Run create_table.py to generate params, PDBs, and alignment CSV."""
    print("\n" + "=" * 80)
    print("STEP 1: Creating alignment table from SDF")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    create_table_script = os.path.join(script_dir, "create_table.py")

    if not os.path.exists(create_table_script):
        raise FileNotFoundError(f"create_table.py not found at: {create_table_script}")

    cmd = [sys.executable, create_table_script, config_file]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)
    print("✓ Table creation complete")
    return result.returncode == 0


def run_docking_array_task(config_file, array_index, mode="glycine"):
    """Step 2: Run docking for a specific array index."""
    print("\n" + "=" * 80)
    print(f"STEP 2: Running docking for array index {array_index} (mode={mode})")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_map = {
        "glycine": "grade_conformers_glycine_shaved.py",
        "sequence": "grade_conformers_docked_to_sequence_multiple_slurm1.py",
    }

    if mode not in script_map:
        raise ValueError(f"Unsupported docking mode '{mode}'. Expected one of: {', '.join(script_map.keys())}")

    docking_script = os.path.join(script_dir, script_map[mode])

    if not os.path.exists(docking_script):
        raise FileNotFoundError(f"Docking script not found at: {docking_script}")

    cmd = [sys.executable, docking_script, config_file, str(array_index)]
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_time

    print(f"✓ Array index {array_index} complete in {elapsed/60:.2f} minutes")
    return result.returncode == 0


def submit_slurm_workflow(config_file, config, wait=False):
    """Submit docking as SLURM array jobs + dependent clustering job.

    This replicates submit_complete_workflow.sh in Python so the workflow
    can be chained with the design pipeline via:
        python run_docking_workflow.py config.txt --slurm --wait && \\
        python run_design_pipeline.py config.txt --wait

    Args:
        config_file: Path to config file (absolute)
        config: Parsed ConfigParser object
        wait: If True, poll squeue until all jobs complete before returning

    Returns:
        True on success, False on failure
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    array_task_count = int(dpu.cfg_get(config, "grade_conformers", "ArrayTaskCount", "1"))
    array_max = array_task_count - 1
    scratch_root = dpu.cfg_get(config, "DEFAULT", "SCRATCH_ROOT", "/scratch/alpine/ryde3462")
    log_dir = os.path.join(scratch_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    docking_script = os.path.join(script_dir, "submit_docking_workflow.sh")
    clustering_script = os.path.join(script_dir, "run_clustering_only.sh")

    if not os.path.exists(docking_script):
        print(f"ERROR: submit_docking_workflow.sh not found at: {docking_script}")
        return False
    if not os.path.exists(clustering_script):
        print(f"ERROR: run_clustering_only.sh not found at: {clustering_script}")
        return False

    print("\n" + "=" * 80)
    print("SUBMITTING DOCKING VIA SLURM")
    print("=" * 80)
    print(f"Config file: {config_file}")
    print(f"Array tasks: 0-{array_max} (total: {array_task_count})")
    print(f"Log directory: {log_dir}")

    # Submit array job
    print("\nStep 1: Submitting array job...")
    cmd = [
        'sbatch', '--parsable', f'--array=0-{array_max}',
        f'--output={log_dir}/docking_%A_%a.out',
        f'--error={log_dir}/docking_%A_%a.err',
        f'--export=ALL,PIPELINE_SCRIPT_DIR={script_dir}',
        docking_script, config_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print(f"ERROR: Failed to submit array job: {result.stderr}")
        return False

    array_job_id = result.stdout.strip()
    print(f"  Array job submitted: {array_job_id}")

    # Submit clustering job with dependency
    print("Step 2: Submitting clustering job (dependency: afterok:{})...".format(array_job_id))
    cmd = [
        'sbatch', '--parsable', f'--dependency=afterok:{array_job_id}',
        f'--output={log_dir}/clustering_%j.out',
        f'--error={log_dir}/clustering_%j.err',
        f'--export=ALL,PIPELINE_SCRIPT_DIR={script_dir}',
        clustering_script, config_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print(f"ERROR: Failed to submit clustering job: {result.stderr}")
        print(f"Array job {array_job_id} is still running")
        return False

    cluster_job_id = result.stdout.strip()
    print(f"  Clustering job submitted: {cluster_job_id}")

    print(f"\nMonitor: squeue -j {array_job_id},{cluster_job_id}")
    print(f"Logs:    tail -f {log_dir}/docking_{array_job_id}_*.out")

    if not wait:
        print("\nJobs submitted. Use --wait to block until completion.")
        return True

    # Poll squeue until both jobs are done
    all_job_ids = f"{array_job_id},{cluster_job_id}"
    print(f"\nWaiting for docking jobs to complete (polling every 60s)...")
    print(f"  Press Ctrl+C to stop waiting (jobs will continue running).\n")

    try:
        while True:
            result = subprocess.run(
                ['squeue', '-j', all_job_ids, '--noheader'],
                capture_output=True, text=True,
            )
            running = [l for l in result.stdout.strip().split('\n') if l.strip()]
            if not running:
                print(f"  [{time.strftime('%H:%M:%S')}] All docking jobs completed.")
                break
            print(f"  [{time.strftime('%H:%M:%S')}] {len(running)} job(s) still running...")
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n  Stopped waiting. Jobs still running: squeue -j {all_job_ids}")
        return True

    # Verify clustered output exists
    output_dir = dpu.cfg_get(config, "grade_conformers", "OutputDir", "output")
    clustered_dir = dpu.cfg_get(
        config, "grade_conformers", "ClusteredOutputDir",
        os.path.join(output_dir, "clustered_final"),
    )
    if os.path.exists(clustered_dir):
        pdb_count = len([f for f in os.listdir(clustered_dir) if f.endswith('.pdb')])
        print(f"\nClustered results: {clustered_dir} ({pdb_count} PDBs)")
    else:
        print(f"\nWARNING: Clustered output directory not found: {clustered_dir}")
        print("  Check SLURM logs for errors.")

    return True


def run_clustering(config_file):
    """Step 3: Run cluster_docked_post_array.py to aggregate and cluster all results."""
    print("\n" + "=" * 80)
    print("STEP 3: Clustering docked conformers across all arrays")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_script = os.path.join(script_dir, "cluster_docked_post_array.py")

    if not os.path.exists(cluster_script):
        raise FileNotFoundError(f"cluster_docked_post_array.py not found at: {cluster_script}")

    cmd = [sys.executable, cluster_script, config_file]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)
    print("✓ Clustering complete")
    return result.returncode == 0


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        "config_file",
        help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "glycine", "sequence"],
        default="auto",
        help="Docking mode to run (default: auto, reads from config)"
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="Specific array index to run (for SLURM array jobs). If specified, only runs that index."
    )
    parser.add_argument(
        "--local-arrays",
        type=int,
        default=None,
        help="Run N array tasks locally in sequence (useful for testing without SLURM)"
    )
    parser.add_argument(
        "--skip-create-table",
        action="store_true",
        help="Skip create_table step (assumes CSV/params already exist)"
    )
    parser.add_argument(
        "--skip-docking",
        action="store_true",
        help="Skip docking step (only run create_table and/or clustering)"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip final clustering step (typically used when running individual array tasks via SLURM)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run create_table only and exit (same as --skip-docking --skip-clustering)"
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit docking as SLURM array jobs (instead of running locally)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for SLURM jobs to complete before returning (use with --slurm)"
    )

    args = parser.parse_args(argv)

    # Load config
    config = ConfigParser()
    with open(args.config_file, "r", encoding="utf-8-sig") as handle:
        config.read_file(handle)

    # Resolve mode
    mode = _resolve_mode(args.mode, config)

    # Get array task count from config
    array_task_count = int(dpu.cfg_get(config, "grade_conformers", "ArrayTaskCount", "1"))

    # Prepare-only mode
    if args.prepare_only:
        args.skip_docking = True
        args.skip_clustering = True

    # STEP 1: Create table
    if not args.skip_create_table:
        csv_file_name = dpu.cfg_get(
            config,
            "create_table",
            "CSVFileName",
            dpu.cfg_get(config, "DEFAULT", "CSVFileName")
        )
        dpu.ensure_table_ready(args.config_file, csv_file_name)
        print(f"✓ Table preparation complete: {csv_file_name}")
    else:
        print("Skipping table creation (--skip-create-table)")

    if args.prepare_only:
        print("\n" + "=" * 80)
        print("PREPARATION COMPLETE")
        print("=" * 80)
        return

    # SLURM submission mode: submit array + clustering jobs and optionally wait
    if args.slurm:
        # Convert config_file to absolute path
        config_file_abs = os.path.abspath(args.config_file)
        success = submit_slurm_workflow(config_file_abs, config, wait=args.wait)
        if not success:
            sys.exit(1)
        print("\n" + "=" * 80)
        print("DOCKING WORKFLOW " + ("COMPLETE" if args.wait else "SUBMITTED"))
        print("=" * 80)
        return

    # STEP 2: Run docking (local mode)
    if not args.skip_docking:
        if args.array_index is not None:
            # Run single array index (called by SLURM job)
            run_docking_array_task(args.config_file, args.array_index, mode)

        elif args.local_arrays is not None:
            # Run multiple array indices locally (for testing)
            print(f"\nRunning {args.local_arrays} array tasks locally...")
            for idx in range(args.local_arrays):
                run_docking_array_task(args.config_file, idx, mode)

        else:
            # Single run (array index 0)
            print(f"\nRunning single docking task (array_task_count={array_task_count} from config)")
            if array_task_count > 1:
                print(f"WARNING: Config specifies ArrayTaskCount={array_task_count}, but running single task.")
                print("Consider using --local-arrays N or submitting via SLURM array.")
            run_docking_array_task(args.config_file, 0, mode)
    else:
        print("Skipping docking (--skip-docking)")

    # STEP 3: Cluster results
    if not args.skip_clustering:
        run_clustering(args.config_file)
    else:
        print("Skipping clustering (--skip-clustering)")

    # Final summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)

    if not args.skip_clustering:
        # Try to show final output location
        output_dir = dpu.cfg_get(config, "grade_conformers", "OutputDir", "output")
        clustered_dir = dpu.cfg_get(
            config,
            "grade_conformers",
            "ClusteredOutputDir",
            os.path.join(output_dir, "clustered_final")
        )
        print(f"\nFinal clustered results should be in: {clustered_dir}")
        if os.path.exists(clustered_dir):
            pdb_count = len([f for f in os.listdir(clustered_dir) if f.endswith('.pdb')])
            print(f"  Found {pdb_count} clustered PDB files")
        print(f"\nCheck cluster_summary.csv in the clustered directory for details.")


if __name__ == "__main__":
    main(sys.argv[1:])
