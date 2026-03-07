#!/bin/bash
#SBATCH --job-name=zscore_orch
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=ucb472_asc2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/alpine/ryde3462/expansion/ligandmpnn/zscore_orchestrator_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ryde3462@colorado.edu
# ============================================================================
# SLURM wrapper for Z-score guided expansion orchestrator
# ============================================================================
#
# Runs the orchestrator as a persistent SLURM job so it doesn't depend
# on an OnDemand session staying alive. Uses minimal resources (1 CPU)
# and just polls squeue between phases.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   sbatch slurm/submit_zscore_expansion.sh 5 7        # rounds 5-7, all ligands
#   sbatch slurm/submit_zscore_expansion.sh 5 7 ca     # rounds 5-7, CA only
#
# Monitor:
#   squeue -u $USER -n zscore_orch
#   tail -f /scratch/alpine/ryde3462/expansion/ligandmpnn/zscore_orchestrator_*.log
#
# ============================================================================

cd "$SLURM_SUBMIT_DIR"

# Pass through all arguments to the orchestrator
bash slurm/run_zscore_expansion.sh "$@"
