#!/bin/bash
#SBATCH --job-name=score_lig
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=ucb472_asc2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alpine/ryde3462/expansion/ligandmpnn/score_%x_%j.log
# ============================================================================
# Score all Boltz predictions for a single ligand (parallelizable)
# ============================================================================
#
# Builds boltz_scored.csv for one ligand from initial + expansion rounds.
# Run one per ligand in parallel, then start the expansion orchestrator.
#
# Usage:
#   sbatch --job-name=score_ca   slurm/submit_score_ligand.sh ca
#   sbatch --job-name=score_cdca slurm/submit_score_ligand.sh cdca
#   sbatch --job-name=score_dca  slurm/submit_score_ligand.sh dca
#
# Then after all complete:
#   sbatch --dependency=afterok:JOB1:JOB2:JOB3 slurm/submit_zscore_expansion.sh 6 9
#
# ============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm/submit_score_ligand.sh <ligand>"
    exit 1
fi

LIG="${1,,}"

cd "$SLURM_SUBMIT_DIR"

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
EXPANSION_ROOT="/scratch/alpine/ryde3462/expansion/ligandmpnn"
INITIAL_BOLTZ_ROOT="/scratch/alpine/ryde3462/boltz_bile_acids"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"

module load anaconda
source activate boltz_env

LIG_DIR="${EXPANSION_ROOT}/${LIG}"
SCORED="${LIG_DIR}/boltz_scored.csv"

# Collect ALL boltz_output directories (initial + expansion rounds)
BOLTZ_DIRS=()

# Initial predictions
INITIAL="${INITIAL_BOLTZ_ROOT}/output_${LIG}_binary"
if [ -d "$INITIAL" ]; then
    BOLTZ_DIRS+=("$INITIAL")
    echo "Including initial: $INITIAL"
fi

# Expansion rounds
for rd in "${LIG_DIR}"/round_*/boltz_output; do
    if [ -d "$rd" ]; then
        BOLTZ_DIRS+=("$rd")
    fi
done

echo "${LIG^^}: scoring ${#BOLTZ_DIRS[@]} directories..."

python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${BOLTZ_DIRS[@]}" \
    --ref-pdb "${REF_PDB}" \
    --out "${SCORED}"

# Write marker so incremental scoring knows we're up to date
MAX_ROUND=0
for rd in "${LIG_DIR}"/round_*/boltz_output; do
    [ -d "$rd" ] || continue
    rn=$(basename "$(dirname "$rd")" | sed 's/round_//')
    [ "$rn" -gt "$MAX_ROUND" ] && MAX_ROUND="$rn"
done
echo "$MAX_ROUND" > "${LIG_DIR}/.last_scored_round"

NROWS=$(tail -n +2 "$SCORED" | wc -l)
echo ""
echo "${LIG^^}: scored ${NROWS} designs → ${SCORED}"
echo "Marker: .last_scored_round = ${MAX_ROUND}"
