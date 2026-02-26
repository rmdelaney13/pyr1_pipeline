#!/bin/bash
# ============================================================================
# Master workflow: Aggregate LCA results + prepare & submit GLCA / LCA-3-S
# ============================================================================
#
# Run this interactively on an Alpine login node (NOT as a SLURM job).
# It will:
#   1. Aggregate existing LCA binary Boltz2 results
#   2. Create GLCA and LCA-3-S tier CSVs (with cross-reference check)
#   3. Generate Boltz YAML inputs for both ligands
#   4. Submit SLURM array jobs for both
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/run_boltz_lca_conjugates.sh
#
# Prerequisites:
#   - boltz_env conda environment
#   - Existing LCA binary results in output_tier1_binary + output_tier4_binary
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"

# Existing LCA binary outputs (two directories: binders + non-binders)
LCA_OUTPUT_TIER1="${SCRATCH}/output_tier1_binary"
LCA_OUTPUT_TIER4="${SCRATCH}/output_tier4_binary"

# Reference PDB with conserved water
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"

# Tier CSV paths
TIER1="${PROJECT_ROOT}/ml_modelling/data/tiers/tier1_strong_binders.csv"
TIER4="${PROJECT_ROOT}/ml_modelling/data/tiers/tier4_LCA_screen.csv"

# Output directories (follows existing naming convention)
GLCA_YAML_DIR="${SCRATCH}/inputs_glca_prod"
LCA3S_YAML_DIR="${SCRATCH}/inputs_lca3s_prod"
GLCA_OUTPUT_DIR="${SCRATCH}/output_glca_binary"
LCA3S_OUTPUT_DIR="${SCRATCH}/output_lca3s_binary"

# Aggregated results
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"

# Boltz settings
DIFFUSION_SAMPLES=5   # binary mode
BATCH_SIZE=20
# MSA generated on-the-fly by Boltz (--use_msa_server, max depth 32 in submit_boltz.sh)

# ── Step 0: Activate environment ──────────────────────────────────────────
echo "============================================"
echo "Step 0: Activating environment"
echo "============================================"

module load anaconda
source activate boltz_env

# ── Step 1: Aggregate existing LCA binary results ─────────────────────────
echo ""
echo "============================================"
echo "Step 1: Aggregate existing LCA binary results"
echo "============================================"

mkdir -p "${RESULTS_DIR}"

LCA_RESULTS="${RESULTS_DIR}/boltz_lca_binary_results.csv"

python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${LCA_OUTPUT_TIER1}" "${LCA_OUTPUT_TIER4}" \
    --ref-pdb "${REF_PDB}" \
    --out "${LCA_RESULTS}"

echo "LCA results written to: ${LCA_RESULTS}"

# ── Step 2: Create GLCA and LCA-3-S tier CSVs ─────────────────────────────
echo ""
echo "============================================"
echo "Step 2: Create GLCA and LCA-3-S CSVs"
echo "============================================"

CONJUGATE_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"

python "${PROJECT_ROOT}/scripts/prepare_boltz_lca_conjugates.py" \
    --tier1 "${TIER1}" \
    --tier4 "${TIER4}" \
    --out-dir "${CONJUGATE_DIR}" \
    --n-nonbinders 500 \
    --seed 42

echo ""
echo "CSVs created in: ${CONJUGATE_DIR}"

# ── Step 3: Generate Boltz YAML inputs ─────────────────────────────────────
# YAMLs omit msa: line (--msa-server); Boltz generates MSAs via --use_msa_server
echo ""
echo "============================================"
echo "Step 3: Generate Boltz YAML inputs"
echo "============================================"

# GlycoLCA YAMLs
echo "--- GlycoLCA ---"
python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
    "${CONJUGATE_DIR}/boltz_glca_binary.csv" \
    --out-dir "${GLCA_YAML_DIR}" \
    --mode binary \
    --msa-server \
    --affinity

GLCA_MANIFEST="${GLCA_YAML_DIR}/manifest.txt"
GLCA_TOTAL=$(wc -l < "${GLCA_MANIFEST}")
GLCA_ARRAY_MAX=$(( (GLCA_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
echo "GlycoLCA: ${GLCA_TOTAL} YAMLs -> array=0-${GLCA_ARRAY_MAX} (batch=${BATCH_SIZE})"

# LCA-3-S YAMLs
echo ""
echo "--- LCA-3-S ---"
python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
    "${CONJUGATE_DIR}/boltz_lca3s_binary.csv" \
    --out-dir "${LCA3S_YAML_DIR}" \
    --mode binary \
    --msa-server \
    --affinity

LCA3S_MANIFEST="${LCA3S_YAML_DIR}/manifest.txt"
LCA3S_TOTAL=$(wc -l < "${LCA3S_MANIFEST}")
LCA3S_ARRAY_MAX=$(( (LCA3S_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
echo "LCA-3-S: ${LCA3S_TOTAL} YAMLs -> array=0-${LCA3S_ARRAY_MAX} (batch=${BATCH_SIZE})"

# ── Step 4: Submit SLURM array jobs ────────────────────────────────────────
# Pass --use_msa_server as extra flag (5th+ arg to submit_boltz.sh)
echo ""
echo "============================================"
echo "Step 4: Submit SLURM jobs"
echo "============================================"

SUBMIT_SCRIPT="${PROJECT_ROOT}/slurm/submit_boltz.sh"

echo "Submitting GlycoLCA (${GLCA_TOTAL} predictions)..."
GLCA_JOB=$(sbatch --array=0-${GLCA_ARRAY_MAX} \
    --job-name=boltz_glca \
    "${SUBMIT_SCRIPT}" \
    "${GLCA_MANIFEST}" "${GLCA_OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
    --use_msa_server \
    | awk '{print $NF}')
echo "  GlycoLCA job: ${GLCA_JOB}"

echo "Submitting LCA-3-S (${LCA3S_TOTAL} predictions)..."
LCA3S_JOB=$(sbatch --array=0-${LCA3S_ARRAY_MAX} \
    --job-name=boltz_lca3s \
    "${SUBMIT_SCRIPT}" \
    "${LCA3S_MANIFEST}" "${LCA3S_OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
    --use_msa_server \
    | awk '{print $NF}')
echo "  LCA-3-S job: ${LCA3S_JOB}"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Jobs submitted:"
echo "  GlycoLCA:  ${GLCA_JOB} (array=0-${GLCA_ARRAY_MAX}, ${GLCA_TOTAL} predictions)"
echo "  LCA-3-S:   ${LCA3S_JOB} (array=0-${LCA3S_ARRAY_MAX}, ${LCA3S_TOTAL} predictions)"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After completion, aggregate results with:"
echo "  bash ${PROJECT_ROOT}/slurm/aggregate_boltz_lca_all.sh"
