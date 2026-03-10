#!/bin/bash
#SBATCH --job-name=CDCA_resume
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=cdca_resume_%j.out
#SBATCH --error=cdca_resume_%j.err
#
# Resume orchestrator: waits for an existing Rosetta SLURM array job to finish,
# then runs aggregate → filter → FASTA → Boltz YAML → submit Boltz GPU jobs.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   sbatch slurm/run_cdca_resume_filter_to_boltz.sh <ROSETTA_JOB_ID>
#
# Example:
#   sbatch slurm/run_cdca_resume_filter_to_boltz.sh 24610962
#

cd "$SLURM_SUBMIT_DIR"

set -eo pipefail

PIPE_ROOT=/projects/ryde3462/software/pyr1_pipeline
CONFIG=${PIPE_ROOT}/campaigns/CDCA/config.txt
SCRATCH=/scratch/alpine/ryde3462/CDCA

module load anaconda
conda activate ligand_alignment

set -u

ROSETTA_JOB_ID="${1:?ERROR: Must provide Rosetta SLURM job ID as first argument}"

echo "============================================="
echo "CDCA Resume Orchestrator (Filter → Boltz)"
echo "Started: $(date)"
echo "Config:  ${CONFIG}"
echo "Waiting for Rosetta job: ${ROSETTA_JOB_ID}"
echo "============================================="

# Poll until the Rosetta array job finishes
echo ""
echo "Waiting for Rosetta job ${ROSETTA_JOB_ID} to complete..."
while squeue -j "${ROSETTA_JOB_ID}" --noheader 2>/dev/null | grep -q .; do
    RUNNING=$(squeue -j "${ROSETTA_JOB_ID}" --noheader 2>/dev/null | wc -l)
    echo "  [$(date '+%H:%M:%S')] ${RUNNING} task(s) still running..."
    sleep 60
done
echo "Rosetta job ${ROSETTA_JOB_ID} completed at $(date)"

# Run aggregate → filter → FASTA → Boltz YAMLs (skip MPNN & Rosetta)
echo ""
echo "Running pipeline from aggregate stage onward..."
python ${PIPE_ROOT}/design/scripts/run_design_pipeline.py ${CONFIG} \
    --rosetta-to-af3 \
    --skip-af3-submit \
    --skip-af3-analyze

echo ""
echo "============================================="
echo "Pipeline stages complete. Submitting Boltz jobs..."
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
