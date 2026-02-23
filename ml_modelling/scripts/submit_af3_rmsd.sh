#!/usr/bin/env bash
#SBATCH --job-name=af3_rmsd
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=4G
# --time and --array are set dynamically by the orchestrator
# --output is set dynamically by the orchestrator

# Usage: sbatch --array=0-N --time=HH:MM:00 --output=... submit_af3_rmsd.sh MANIFEST PAIRS_PER_TASK

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || true

MANIFEST="${1:?Usage: submit_af3_rmsd.sh MANIFEST PAIRS_PER_TASK}"
PAIRS_PER_TASK="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[$(date)] AF3 RMSD task ${SLURM_ARRAY_TASK_ID} — ${PAIRS_PER_TASK} pairs from ${MANIFEST}"

python "${SCRIPT_DIR}/compute_af3_rmsd.py" \
    --manifest "${MANIFEST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --pairs-per-task "${PAIRS_PER_TASK}"

echo "[$(date)] AF3 RMSD task ${SLURM_ARRAY_TASK_ID} complete"
