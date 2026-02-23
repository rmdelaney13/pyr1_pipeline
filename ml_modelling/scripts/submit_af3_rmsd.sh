#!/usr/bin/env bash
#SBATCH --job-name=af3_rmsd
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=4G
# --time and --array are set dynamically by the orchestrator
# --output is set dynamically by the orchestrator

# Usage: sbatch --array=0-N --time=HH:MM:00 --output=... submit_af3_rmsd.sh MANIFEST PAIRS_PER_TASK COMPUTE_SCRIPT
#
# COMPUTE_SCRIPT is the absolute path to compute_af3_rmsd.py, passed by the
# orchestrator because $0 in SLURM points to a spool copy, not the original.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || true

MANIFEST="${1:?Usage: submit_af3_rmsd.sh MANIFEST PAIRS_PER_TASK COMPUTE_SCRIPT}"
PAIRS_PER_TASK="${2:-1}"
COMPUTE_SCRIPT="${3:?Must provide absolute path to compute_af3_rmsd.py}"

echo "[$(date)] AF3 RMSD task ${SLURM_ARRAY_TASK_ID} — ${PAIRS_PER_TASK} pairs from ${MANIFEST}"

python "${COMPUTE_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --pairs-per-task "${PAIRS_PER_TASK}"

RC=$?
echo "[$(date)] AF3 RMSD task ${SLURM_ARRAY_TASK_ID} complete (exit code ${RC})"
exit $RC
