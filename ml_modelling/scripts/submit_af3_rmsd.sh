#!/usr/bin/env bash
#SBATCH --job-name=af3_rmsd
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=4G
# --time and --array are set dynamically by the orchestrator
# --output is set dynamically by the orchestrator

# Usage: sbatch --array=0-N --time=HH:MM:00 --output=... submit_af3_rmsd.sh \
#        MANIFEST PAIRS_PER_TASK COMPUTE_SCRIPT [REF_MODEL] [--recompute]
#
# COMPUTE_SCRIPT is the absolute path to compute_af3_rmsd.py, passed by the
# orchestrator because $0 in SLURM points to a spool copy, not the original.
# REF_MODEL (optional) is the reference PDB for H-bond acceptor distance.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || true

MANIFEST="${1:?Usage: submit_af3_rmsd.sh MANIFEST PAIRS_PER_TASK COMPUTE_SCRIPT [REF_MODEL] [--recompute]}"
PAIRS_PER_TASK="${2:-1}"
COMPUTE_SCRIPT="${3:?Must provide absolute path to compute_af3_rmsd.py}"
REF_MODEL="${4:-}"
RECOMPUTE="${5:-}"

echo "[$(date)] AF3 RMSD task ${SLURM_ARRAY_TASK_ID} — ${PAIRS_PER_TASK} pairs from ${MANIFEST}"

EXTRA_ARGS=""
if [ -n "$REF_MODEL" ] && [ "$REF_MODEL" != "--recompute" ]; then
    EXTRA_ARGS="--ref-model ${REF_MODEL}"
fi

# Check both positional arg 4 and 5 for --recompute flag
if [ "$REF_MODEL" = "--recompute" ] || [ "$RECOMPUTE" = "--recompute" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --recompute"
fi

python "${COMPUTE_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --pairs-per-task "${PAIRS_PER_TASK}" \
    ${EXTRA_ARGS}

RC=$?
echo "[$(date)] AF3 RMSD task ${SLURM_ARRAY_TASK_ID} complete (exit code ${RC})"
exit $RC
