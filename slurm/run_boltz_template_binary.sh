#!/bin/bash
# ============================================================================
# Binary template-based Boltz2 predictions for LCA, GLCA, and LCA-3-S
# ============================================================================
#
# Run interactively on an Alpine login node (NOT as a SLURM job).
# Uses MAXIT-converted PYR1 template CIF with msa: empty (single-sequence).
#
# Generates YAMLs and submits SLURM array jobs for 3 ligands:
#   - LCA:     52 binders + ~500 non-binders (same non-binder pool as conjugates)
#   - GLCA:    18 binders + ~500 non-binders
#   - LCA-3-S: 18 binders + ~500 non-binders
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/run_boltz_template_binary.sh
#
# Prerequisites:
#   - boltz_env conda environment
#   - MAXIT-converted template CIF in structures/templates/
#   - Tier CSVs (tier1_strong_binders.csv, tier4_LCA_screen.csv)
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"

# Template CIF (MAXIT-converted, has _entity_poly + _entity_poly_seq)
TEMPLATE="${PROJECT_ROOT}/structures/templates/Pyr1_LCA_mutant_template_converted.cif"

# Conjugate CSVs (all 3 ligands share the same ~500 non-binder pool)
CONJUGATE_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"
LCA_CSV="${CONJUGATE_DIR}/boltz_lca_binary.csv"
GLCA_CSV="${CONJUGATE_DIR}/boltz_glca_binary.csv"
LCA3S_CSV="${CONJUGATE_DIR}/boltz_lca3s_binary.csv"

# YAML input directories
LCA_YAML_DIR="${SCRATCH}/inputs_lca_template"
GLCA_YAML_DIR="${SCRATCH}/inputs_glca_template"
LCA3S_YAML_DIR="${SCRATCH}/inputs_lca3s_template"

# Output directories
LCA_OUTPUT_DIR="${SCRATCH}/output_lca_binary_template"
GLCA_OUTPUT_DIR="${SCRATCH}/output_glca_binary_template"
LCA3S_OUTPUT_DIR="${SCRATCH}/output_lca3s_binary_template"

# Boltz settings
DIFFUSION_SAMPLES=5
BATCH_SIZE=20
SUBMIT_SCRIPT="${PROJECT_ROOT}/slurm/submit_boltz.sh"

# ── Step 0: Activate environment ──────────────────────────────────────────
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi

# ── Step 1: Verify template ────────────────────────────────────────────────
echo "============================================"
echo "Step 1: Verifying template"
echo "============================================"

if [ ! -f "${TEMPLATE}" ]; then
    echo "ERROR: Template CIF not found at ${TEMPLATE}"
    exit 1
fi
echo "Template: ${TEMPLATE}"

# ── Step 2: Generate CSVs (LCA + GLCA + LCA-3-S, shared non-binder pool) ─
echo ""
echo "============================================"
echo "Step 2: Generate prediction CSVs"
echo "============================================"

TIER1="${PROJECT_ROOT}/ml_modelling/data/tiers/tier1_strong_binders.csv"
TIER4="${PROJECT_ROOT}/ml_modelling/data/tiers/tier4_LCA_screen.csv"

python "${PROJECT_ROOT}/scripts/prepare_boltz_lca_conjugates.py" \
    --tier1 "${TIER1}" \
    --tier4 "${TIER4}" \
    --out-dir "${CONJUGATE_DIR}" \
    --n-nonbinders 500 \
    --seed 42

for F in "${LCA_CSV}" "${GLCA_CSV}" "${LCA3S_CSV}"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: CSV not generated: $F"
        exit 1
    fi
done
echo ""
echo "LCA:      ${LCA_CSV}"
echo "GLCA:     ${GLCA_CSV}"
echo "LCA-3-S:  ${LCA3S_CSV}"

# ── Helper function ───────────────────────────────────────────────────────
generate_and_submit() {
    local LABEL="$1"
    local CSV_ARGS="$2"      # space-separated CSV paths
    local YAML_DIR="$3"
    local OUTPUT_DIR="$4"
    local EXTRA_ARGS="${5:-}"  # optional: --ligand-filter etc.

    echo ""
    echo "============================================"
    echo "${LABEL}: Generating YAMLs"
    echo "============================================"

    # shellcheck disable=SC2086
    python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
        ${CSV_ARGS} \
        --out-dir "${YAML_DIR}" \
        --mode binary \
        --template "${TEMPLATE}" \
        --affinity \
        ${EXTRA_ARGS}

    local MANIFEST="${YAML_DIR}/manifest.txt"
    local TOTAL
    TOTAL=$(wc -l < "${MANIFEST}")
    local ARRAY_MAX=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))

    echo "${LABEL}: ${TOTAL} YAMLs -> array=0-${ARRAY_MAX} (batch=${BATCH_SIZE})"

    echo ""
    echo "${LABEL}: Submitting SLURM job"
    local JOB
    JOB=$(sbatch --array=0-${ARRAY_MAX} \
        --job-name="boltz_${LABEL,,}" \
        "${SUBMIT_SCRIPT}" \
        "${MANIFEST}" "${OUTPUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
        | awk '{print $NF}')
    echo "  ${LABEL} job: ${JOB} (array=0-${ARRAY_MAX}, ${TOTAL} predictions)"

    # Export for summary
    eval "${LABEL}_JOB=${JOB}"
    eval "${LABEL}_TOTAL=${TOTAL}"
    eval "${LABEL}_ARRAY_MAX=${ARRAY_MAX}"
}

# ── Step 3: Generate YAMLs and submit ─────────────────────────────────────

# LCA: same non-binder pool as conjugates
generate_and_submit "LCA" "${LCA_CSV}" "${LCA_YAML_DIR}" "${LCA_OUTPUT_DIR}"

# GLCA: conjugate CSV (already has binders + non-binders)
generate_and_submit "GLCA" "${GLCA_CSV}" "${GLCA_YAML_DIR}" "${GLCA_OUTPUT_DIR}"

# LCA-3-S: conjugate CSV
generate_and_submit "LCA3S" "${LCA3S_CSV}" "${LCA3S_YAML_DIR}" "${LCA3S_OUTPUT_DIR}"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "ALL DONE — Binary template predictions"
echo "============================================"
echo ""
echo "Jobs submitted:"
echo "  LCA:     ${LCA_JOB} (${LCA_TOTAL} predictions)"
echo "  GLCA:    ${GLCA_JOB} (${GLCA_TOTAL} predictions)"
echo "  LCA-3-S: ${LCA3S_JOB} (${LCA3S_TOTAL} predictions)"
echo ""
echo "Template: ${TEMPLATE}"
echo "Mode: binary (diffusion_samples=${DIFFUSION_SAMPLES}, msa: empty)"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "Output directories:"
echo "  ${LCA_OUTPUT_DIR}"
echo "  ${GLCA_OUTPUT_DIR}"
echo "  ${LCA3S_OUTPUT_DIR}"
