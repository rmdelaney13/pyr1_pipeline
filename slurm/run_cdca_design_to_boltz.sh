#!/bin/bash
#SBATCH --job-name=CDCA_orchestrate
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=cdca_orchestrate_%j.out
#SBATCH --error=cdca_orchestrate_%j.err
#
# Orchestrator job: runs the full CDCA design pipeline on a CPU node,
# then submits Boltz GPU jobs at the end.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   sbatch slurm/run_cdca_design_to_boltz.sh
#
# This single job handles:
#   1. MPNN (submits SLURM array, waits)
#   2. Rosetta relax (submits SLURM array, waits)
#   3. Score aggregation
#   4. Filtering (oxygen-class quotas)
#   5. FASTA generation
#   6. Boltz YAML generation (with patched MSA)
#   7. Submits Boltz GPU array jobs

cd "$SLURM_SUBMIT_DIR"

set -eo pipefail

PIPE_ROOT=/projects/ryde3462/software/pyr1_pipeline
CONFIG=${PIPE_ROOT}/campaigns/CDCA/config.txt
SCRATCH=/scratch/alpine/ryde3462/CDCA

module load anaconda
conda activate ligand_alignment

set -u

echo "============================================="
echo "CDCA Design Pipeline Orchestrator"
echo "Started: $(date)"
echo "Config: ${CONFIG}"
echo "============================================="

# Clear previous iteration output
echo ""
echo "Clearing previous iteration_1 output..."
rm -rf ${SCRATCH}/design/iteration_1
echo "Done."

# Run pipeline stages 1-6 (MPNN → Rosetta → Aggregate → Filter → FASTA → Boltz YAMLs)
echo ""
echo "Running design pipeline with --wait..."
python ${PIPE_ROOT}/design/scripts/run_design_pipeline.py ${CONFIG} --wait

echo ""
echo "============================================="
echo "Pipeline complete. Submitting Boltz jobs..."
echo "============================================="

# Submit Boltz binary predictions
BOLTZ_DIR=${SCRATCH}/design/boltz_inputs
MANIFEST=${BOLTZ_DIR}/manifest.txt

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found at ${MANIFEST}"
    echo "Check pipeline output above for errors."
    exit 1
fi

N=$(wc -l < "$MANIFEST")
if [ "$N" -eq 0 ]; then
    echo "ERROR: Manifest is empty"
    exit 1
fi

BATCH_SIZE=20
ARRAY_MAX=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))

echo "Manifest: ${MANIFEST}"
echo "Total YAMLs: ${N}"
echo "Batch size: ${BATCH_SIZE}"
echo "Array range: 0-${ARRAY_MAX}"

BOLTZ_JOB_ID=$(sbatch --parsable \
    --array=0-${ARRAY_MAX} \
    ${PIPE_ROOT}/slurm/submit_boltz.sh \
    "${MANIFEST}" \
    "${SCRATCH}/design/boltz_output" \
    ${BATCH_SIZE} \
    5 \
    --max_msa_seqs 128)

echo "Submitted Boltz job: ${BOLTZ_JOB_ID}"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Finished: $(date)"
