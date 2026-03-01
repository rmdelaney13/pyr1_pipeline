#!/bin/bash
# ============================================================================
# Master workflow: Prepare & submit ternary Boltz2 predictions for
# LCA, GLCA, and LCA-3-S training data (PYR1 + ligand + HAB1)
# ============================================================================
#
# Run this interactively on an Alpine login node (NOT as a SLURM job).
# It will:
#   1. Extract standalone HAB1 MSA from WT ternary prediction output
#   2. Generate LCA binary CSV if missing (plain LCA binders + non-binders)
#   3. Generate ternary YAML inputs for LCA, GLCA, and LCA-3-S
#   4. Submit SLURM array jobs for all three ligands
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/run_boltz_lca_ternary.sh
#
# Prerequisites:
#   - boltz_env conda environment (with confidence patch applied)
#   - Completed WT ternary prediction at $WT_TERNARY_DIR
#   - Existing conjugate CSVs in $CONJUGATE_DIR
#   - WT PYR1 MSA from binary predictions
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"

# WT ternary prediction (source of HAB1 MSA)
WT_TERNARY_DIR="${SCRATCH}/wt_ternary"
WT_TERNARY_MSA_DIR="${WT_TERNARY_DIR}/boltz_results_pyr1_wt_lca_hab1/msa"

# WT PYR1 MSA (from original binary WT prediction, used as base for per-variant patching)
WT_MSA="${SCRATCH}/wt_prediction/boltz_results_pyr1_wt_lca/msa/pyr1_wt_lca_unpaired_tmp_env/uniref.a3m"

# Pre-computed standalone HAB1 MSA (extracted from WT ternary)
HAB1_MSA_DIR="${SCRATCH}/hab1_msa"
HAB1_MSA="${HAB1_MSA_DIR}/hab1.a3m"

# Reference PDB with conserved water
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"

# Tier CSV paths
TIER1="${PROJECT_ROOT}/ml_modelling/data/tiers/tier1_strong_binders.csv"
TIER4="${PROJECT_ROOT}/ml_modelling/data/tiers/tier4_LCA_screen.csv"

# Conjugate CSV directory
CONJUGATE_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"

# Ternary YAML input directories
LCA_YAML_DIR="${SCRATCH}/inputs_lca_ternary"
GLCA_YAML_DIR="${SCRATCH}/inputs_glca_ternary"
LCA3S_YAML_DIR="${SCRATCH}/inputs_lca3s_ternary"

# Ternary output directories
LCA_OUTPUT_DIR="${SCRATCH}/output_lca_ternary"
GLCA_OUTPUT_DIR="${SCRATCH}/output_glca_ternary"
LCA3S_OUTPUT_DIR="${SCRATCH}/output_lca3s_ternary"

# Boltz settings
DIFFUSION_SAMPLES=5   # same as binary for quality (confidence patch handles aggregation)
BATCH_SIZE=10          # smaller batch (ternary ~5 min each vs ~1 min binary)

# ── Step 0: Activate environment ──────────────────────────────────────────
echo "============================================"
echo "Step 0: Activating environment"
echo "============================================"

module load anaconda
source activate boltz_env

# ── Step 1: Extract HAB1 MSA ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "Step 1: Extract standalone HAB1 MSA"
echo "============================================"

if [ -f "${HAB1_MSA}" ]; then
    echo "HAB1 MSA already exists: ${HAB1_MSA}"
    echo "  $(wc -l < "${HAB1_MSA}") lines"
else
    # Check WT ternary MSA directory exists
    UNIREF_A3M="${WT_TERNARY_MSA_DIR}/pyr1_wt_lca_hab1_unpaired_tmp_env/uniref.a3m"
    BFD_A3M="${WT_TERNARY_MSA_DIR}/pyr1_wt_lca_hab1_unpaired_tmp_env/bfd.mgnify30.metaeuk30.smag30.a3m"

    if [ ! -f "${UNIREF_A3M}" ]; then
        echo "ERROR: WT ternary MSA not found at ${UNIREF_A3M}"
        echo "Run predict_wt_ternary.sh first to generate WT ternary prediction with MSA."
        exit 1
    fi

    mkdir -p "${HAB1_MSA_DIR}"

    # Extract HAB1 (chain index 2) from multi-chain .a3m files
    # Chain order: 0=PYR1 (A), 1=ligand (B), 2=HAB1 (C)
    EXTRACT_ARGS=("${UNIREF_A3M}")
    if [ -f "${BFD_A3M}" ]; then
        EXTRACT_ARGS+=("${BFD_A3M}")
    fi

    python "${PROJECT_ROOT}/scripts/extract_chain_msa.py" \
        "${EXTRACT_ARGS[@]}" \
        --chain-index 2 \
        --out "${HAB1_MSA}"

    echo "HAB1 MSA extracted to: ${HAB1_MSA}"
fi

# ── Step 2: Verify MSA files ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "Step 2: Verify MSA files"
echo "============================================"

if [ ! -f "${WT_MSA}" ]; then
    echo "ERROR: WT PYR1 MSA not found at ${WT_MSA}"
    exit 1
fi
echo "PYR1 WT MSA: ${WT_MSA} ($(wc -l < "${WT_MSA}") lines)"

if [ ! -f "${HAB1_MSA}" ]; then
    echo "ERROR: HAB1 MSA not found at ${HAB1_MSA}"
    exit 1
fi
echo "HAB1 MSA:    ${HAB1_MSA} ($(wc -l < "${HAB1_MSA}") lines)"

# ── Step 3: Generate LCA binary CSV if missing ──────────────────────────
echo ""
echo "============================================"
echo "Step 3: Check/generate conjugate CSVs"
echo "============================================"

LCA_CSV="${CONJUGATE_DIR}/boltz_lca_binary.csv"
GLCA_CSV="${CONJUGATE_DIR}/boltz_glca_binary.csv"
LCA3S_CSV="${CONJUGATE_DIR}/boltz_lca3s_binary.csv"

if [ ! -f "${LCA_CSV}" ]; then
    echo "LCA binary CSV not found — generating..."
    python "${PROJECT_ROOT}/scripts/prepare_boltz_lca_conjugates.py" \
        --tier1 "${TIER1}" \
        --tier4 "${TIER4}" \
        --out-dir "${CONJUGATE_DIR}" \
        --n-nonbinders 500 \
        --seed 42
    echo ""
else
    echo "LCA CSV:    ${LCA_CSV} ($(wc -l < "${LCA_CSV}") lines)"
fi

if [ ! -f "${GLCA_CSV}" ]; then
    echo "ERROR: GLCA CSV not found at ${GLCA_CSV}"
    echo "Run: bash slurm/run_boltz_lca_conjugates.sh (step 2)"
    exit 1
fi
echo "GLCA CSV:   ${GLCA_CSV} ($(wc -l < "${GLCA_CSV}") lines)"

if [ ! -f "${LCA3S_CSV}" ]; then
    echo "ERROR: LCA-3-S CSV not found at ${LCA3S_CSV}"
    echo "Run: bash slurm/run_boltz_lca_conjugates.sh (step 2)"
    exit 1
fi
echo "LCA-3-S CSV: ${LCA3S_CSV} ($(wc -l < "${LCA3S_CSV}") lines)"

# ── Step 4: Generate ternary YAML inputs ─────────────────────────────────
echo ""
echo "============================================"
echo "Step 4: Generate ternary YAML inputs"
echo "============================================"

# Plain LCA ternary YAMLs
echo ""
echo "--- Plain LCA (ternary) ---"
python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
    "${LCA_CSV}" \
    --out-dir "${LCA_YAML_DIR}" \
    --mode ternary \
    --msa "${WT_MSA}" \
    --hab1-msa "${HAB1_MSA}"

LCA_MANIFEST="${LCA_YAML_DIR}/manifest.txt"
LCA_TOTAL=$(wc -l < "${LCA_MANIFEST}")
LCA_ARRAY_MAX=$(( (LCA_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
echo "LCA ternary: ${LCA_TOTAL} YAMLs -> array=0-${LCA_ARRAY_MAX} (batch=${BATCH_SIZE})"

# GlycoLCA ternary YAMLs
echo ""
echo "--- GlycoLCA (ternary) ---"
python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
    "${GLCA_CSV}" \
    --out-dir "${GLCA_YAML_DIR}" \
    --mode ternary \
    --msa "${WT_MSA}" \
    --hab1-msa "${HAB1_MSA}"

GLCA_MANIFEST="${GLCA_YAML_DIR}/manifest.txt"
GLCA_TOTAL=$(wc -l < "${GLCA_MANIFEST}")
GLCA_ARRAY_MAX=$(( (GLCA_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
echo "GLCA ternary: ${GLCA_TOTAL} YAMLs -> array=0-${GLCA_ARRAY_MAX} (batch=${BATCH_SIZE})"

# LCA-3-S ternary YAMLs
echo ""
echo "--- LCA-3-S (ternary) ---"
python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
    "${LCA3S_CSV}" \
    --out-dir "${LCA3S_YAML_DIR}" \
    --mode ternary \
    --msa "${WT_MSA}" \
    --hab1-msa "${HAB1_MSA}"

LCA3S_MANIFEST="${LCA3S_YAML_DIR}/manifest.txt"
LCA3S_TOTAL=$(wc -l < "${LCA3S_MANIFEST}")
LCA3S_ARRAY_MAX=$(( (LCA3S_TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))
echo "LCA-3-S ternary: ${LCA3S_TOTAL} YAMLs -> array=0-${LCA3S_ARRAY_MAX} (batch=${BATCH_SIZE})"

# ── Step 5: Submit SLURM array jobs ──────────────────────────────────────
echo ""
echo "============================================"
echo "Step 5: Submit SLURM jobs"
echo "============================================"

SUBMIT_SCRIPT="${PROJECT_ROOT}/slurm/submit_boltz.sh"

echo "Submitting LCA ternary (${LCA_TOTAL} predictions)..."
LCA_JOB=$(sbatch --array=0-${LCA_ARRAY_MAX} \
    --job-name=boltz_lca_tern \
    "${SUBMIT_SCRIPT}" \
    "${LCA_MANIFEST}" "${LCA_OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
    | awk '{print $NF}')
echo "  LCA ternary job: ${LCA_JOB}"

echo "Submitting GLCA ternary (${GLCA_TOTAL} predictions)..."
GLCA_JOB=$(sbatch --array=0-${GLCA_ARRAY_MAX} \
    --job-name=boltz_glca_tern \
    "${SUBMIT_SCRIPT}" \
    "${GLCA_MANIFEST}" "${GLCA_OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
    | awk '{print $NF}')
echo "  GLCA ternary job: ${GLCA_JOB}"

echo "Submitting LCA-3-S ternary (${LCA3S_TOTAL} predictions)..."
LCA3S_JOB=$(sbatch --array=0-${LCA3S_ARRAY_MAX} \
    --job-name=boltz_lca3s_tern \
    "${SUBMIT_SCRIPT}" \
    "${LCA3S_MANIFEST}" "${LCA3S_OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
    | awk '{print $NF}')
echo "  LCA-3-S ternary job: ${LCA3S_JOB}"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "ALL DONE — Ternary predictions submitted"
echo "============================================"
echo ""
echo "Jobs submitted:"
echo "  LCA:       ${LCA_JOB} (array=0-${LCA_ARRAY_MAX}, ${LCA_TOTAL} predictions)"
echo "  GLCA:      ${GLCA_JOB} (array=0-${GLCA_ARRAY_MAX}, ${GLCA_TOTAL} predictions)"
echo "  LCA-3-S:   ${LCA3S_JOB} (array=0-${LCA3S_ARRAY_MAX}, ${LCA3S_TOTAL} predictions)"
echo ""
echo "Total: $(( LCA_TOTAL + GLCA_TOTAL + LCA3S_TOTAL )) ternary predictions"
echo "Diffusion samples: ${DIFFUSION_SAMPLES}"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After completion, aggregate results with:"
echo "  bash ${PROJECT_ROOT}/slurm/aggregate_boltz_lca_ternary.sh"
