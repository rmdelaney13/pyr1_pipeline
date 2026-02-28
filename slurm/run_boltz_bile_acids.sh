#!/bin/bash
# ============================================================================
# Master workflow: Prepare & submit Boltz2 binary predictions for bile acids
# (CA, CDCA, UDCA, DCA) — Round 2 design filtering
# ============================================================================
#
# Run this interactively on an Alpine login node (NOT as a SLURM job).
# It will:
#   1. Convert per-ligand FASTA files to tier CSVs
#   2. Generate Boltz YAML inputs with MSA patching + affinity
#   3. Submit SLURM array jobs for each ligand
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/run_boltz_bile_acids.sh
#
# Prerequisites:
#   - boltz_env conda environment
#   - Per-ligand FASTA files at FASTA_DIR
#   - WT PYR1 MSA from previous run (reused from LCA predictions)
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_bile_acids"

# Input FASTA files (one per ligand)
FASTA_DIR="/scratch/alpine/ryde3462/bile_seqs_20260114"
FASTA_FILES=(
    "${FASTA_DIR}/CA_aggregated.fasta"
    "${FASTA_DIR}/CDCA_aggregated.fasta"
    "${FASTA_DIR}/UDCA_aggregated.fasta"
    "${FASTA_DIR}/DCA_aggregated.fasta"
)

# WT PYR1 MSA (reuse from LCA predictions — MSA is protein-only, ligand-independent)
WT_MSA="/scratch/alpine/ryde3462/boltz_lca/wt_prediction/boltz_results_pyr1_wt_lca/msa/pyr1_wt_lca_unpaired_tmp_env/uniref.a3m"

# Boltz settings
DIFFUSION_SAMPLES=5   # binary mode
BATCH_SIZE=20

# Ligands (must match filenames: CA_aggregated.fasta -> ca)
LIGANDS=("ca" "cdca" "udca" "dca")

# ── Step 0: Activate environment ──────────────────────────────────────────
echo "============================================"
echo "Step 0: Activating environment"
echo "============================================"

module load anaconda
source activate boltz_env

# ── Step 1: Convert FASTA files to per-ligand tier CSVs ───────────────────
echo ""
echo "============================================"
echo "Step 1: Convert FASTA files to per-ligand CSVs"
echo "============================================"

CSV_DIR="${SCRATCH}/csvs"
mkdir -p "${CSV_DIR}"

# Verify all FASTA files exist
for FA in "${FASTA_FILES[@]}"; do
    if [ ! -f "${FA}" ]; then
        echo "ERROR: FASTA not found: ${FA}"
        exit 1
    fi
    echo "  Found: ${FA} ($(grep -c '^>' "${FA}") sequences)"
done

python "${PROJECT_ROOT}/scripts/prepare_bile_acid_csvs.py" \
    "${FASTA_FILES[@]}" \
    --out-dir "${CSV_DIR}"

# ── Step 2: Verify WT MSA ────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Step 2: Verify WT MSA"
echo "============================================"

if [ ! -f "${WT_MSA}" ]; then
    echo "ERROR: WT MSA not found at ${WT_MSA}"
    echo "Generate it first with: sbatch slurm/generate_pyr1_msa.sh"
    exit 1
fi
echo "Using WT MSA: ${WT_MSA} ($(wc -l < "${WT_MSA}") lines)"

# ── Step 3+4: Generate YAMLs and submit jobs per ligand ───────────────────
echo ""
echo "============================================"
echo "Step 3: Generate Boltz YAML inputs + submit jobs"
echo "============================================"

SUBMIT_SCRIPT="${PROJECT_ROOT}/slurm/submit_boltz.sh"
SUBMITTED_JOBS=()

for LIG in "${LIGANDS[@]}"; do
    CSV_FILE="${CSV_DIR}/boltz_${LIG}_binary.csv"

    if [ ! -f "${CSV_FILE}" ]; then
        echo "No CSV for ${LIG^^}, skipping"
        continue
    fi

    YAML_DIR="${SCRATCH}/inputs_${LIG}_binary"
    OUTPUT_DIR="${SCRATCH}/output_${LIG}_binary"

    echo ""
    echo "--- ${LIG^^} ---"
    python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
        "${CSV_FILE}" \
        --out-dir "${YAML_DIR}" \
        --mode binary \
        --msa "${WT_MSA}" \
        --affinity

    MANIFEST="${YAML_DIR}/manifest.txt"
    TOTAL=$(wc -l < "${MANIFEST}")
    ARRAY_MAX=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
    echo "${LIG^^}: ${TOTAL} YAMLs -> array=0-${ARRAY_MAX} (batch=${BATCH_SIZE})"

    echo "Submitting ${LIG^^} (${TOTAL} predictions)..."
    JOB_ID=$(sbatch --array=0-${ARRAY_MAX} \
        --job-name=boltz_${LIG} \
        "${SUBMIT_SCRIPT}" \
        "${MANIFEST}" "${OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
        | awk '{print $NF}')
    echo "  ${LIG^^} job: ${JOB_ID}"
    SUBMITTED_JOBS+=("${LIG^^}:${JOB_ID}:${TOTAL}")
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Jobs submitted:"
for entry in "${SUBMITTED_JOBS[@]}"; do
    IFS=':' read -r lig job total <<< "$entry"
    echo "  ${lig}: job ${job} (${total} predictions)"
done
echo ""
echo "Settings: diffusion_samples=${DIFFUSION_SAMPLES}, max_msa_seqs=32 (in submit_boltz.sh), affinity=yes"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After completion, aggregate results with:"
echo "  python ${PROJECT_ROOT}/scripts/analyze_boltz_output.py \\"
for LIG in "${LIGANDS[@]}"; do
    echo "      --binary-dir ${SCRATCH}/output_${LIG}_binary \\"
done
echo "      --ref-pdb ${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb \\"
echo "      --out ${SCRATCH}/boltz_bile_acids_results.csv"
