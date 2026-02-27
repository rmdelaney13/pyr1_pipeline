#!/bin/bash
# ============================================================================
# Deep comparison of MSA vs template Boltz2 binary predictions for LCA
# ============================================================================
# Runs the comprehensive analysis script on the shared LCA variant panel.
#
# Prerequisites:
#   - MSA results:      /scratch/alpine/ryde3462/boltz_lca/results_all_affinity.csv
#   - Template results: ml_modelling/analysis/boltz_LCA/boltz_lca_binary_template_results.csv
#   - Labels:           ml_modelling/data/boltz_lca_conjugates/boltz_lca_binary.csv
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/deep_compare_msa_template.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"

MSA_RESULTS="${SCRATCH}/results_all_affinity.csv"
TMPL_RESULTS="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA/boltz_lca_binary_template_results.csv"
LABELS="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates/boltz_lca_binary.csv"
OUT_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA/msa_vs_template"

# Check conda env
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi

# Verify inputs
for f in "${MSA_RESULTS}" "${TMPL_RESULTS}" "${LABELS}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing input file: $f"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

echo "============================================"
echo "Deep MSA vs Template comparison (LCA)"
echo "============================================"
echo "MSA results:      ${MSA_RESULTS}"
echo "Template results: ${TMPL_RESULTS}"
echo "Labels:           ${LABELS}"
echo "Output:           ${OUT_DIR}"
echo ""

python "${PROJECT_ROOT}/scripts/deep_compare_msa_vs_template.py" \
    --msa-results "${MSA_RESULTS}" \
    --template-results "${TMPL_RESULTS}" \
    --labels "${LABELS}" \
    --ligand LCA \
    --out-dir "${OUT_DIR}"

echo ""
echo "============================================"
echo "Results written to: ${OUT_DIR}/"
echo "============================================"
ls -la "${OUT_DIR}/" 2>/dev/null || echo "(empty)"
