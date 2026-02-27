#!/bin/bash
# ============================================================================
# Cross-ligand scoring analysis for template-based Boltz2 binary predictions
# ============================================================================
# Run after aggregate_boltz_template_binary.sh produces per-ligand CSVs.
#
# 1. Runs quick_boltz_analysis.py (fixed) for each ligand with proper labels
# 2. Runs cross-ligand scoring analysis across all three ligands
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/analyze_boltz_template_scoring.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"
LABELS_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"
SCORING_DIR="${RESULTS_DIR}/scoring"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi

mkdir -p "${SCORING_DIR}"

# ── Quick per-ligand analysis (with proper labels) ──
echo "============================================"
echo "Per-ligand quick analysis (with labels)"
echo "============================================"

for LIGAND in lca glca lca3s; do
    RESULTS="${RESULTS_DIR}/boltz_${LIGAND}_binary_template_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"

    if [ ! -f "${RESULTS}" ]; then
        echo "SKIP: ${RESULTS} not found"
        continue
    fi
    if [ ! -f "${LABELS}" ]; then
        echo "SKIP: ${LABELS} not found"
        continue
    fi

    echo ""
    echo "--- ${LIGAND} (template, with labels) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${RESULTS}" --labels "${LABELS}" 2>/dev/null || true
done

# ── Cross-ligand scoring analysis ──
echo ""
echo "============================================"
echo "Cross-ligand scoring analysis"
echo "============================================"

# Build --data arguments for each available ligand
DATA_ARGS=""
for LIGAND in lca glca lca3s; do
    RESULTS="${RESULTS_DIR}/boltz_${LIGAND}_binary_template_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"

    if [ -f "${RESULTS}" ] && [ -f "${LABELS}" ]; then
        DATA_ARGS="${DATA_ARGS} --data ${LIGAND^^} ${RESULTS} ${LABELS}"
    fi
done

if [ -z "${DATA_ARGS}" ]; then
    echo "ERROR: No result/label CSV pairs found"
    exit 1
fi

# shellcheck disable=SC2086
python "${PROJECT_ROOT}/scripts/analyze_boltz_scoring.py" \
    ${DATA_ARGS} \
    --out-dir "${SCORING_DIR}"

echo ""
echo "============================================"
echo "Results written to: ${SCORING_DIR}/"
echo "============================================"
ls -la "${SCORING_DIR}/"*.csv 2>/dev/null || echo "(no CSVs found)"
