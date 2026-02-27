#!/bin/bash
# ============================================================================
# Intermediate analysis: available Boltz2 binary results
# ============================================================================
# Analyzes the 3 completed runs:
#   - LCA MSA (lca_msa_binary_output/output_tier1_binary + output_tier4_binary)
#   - GLCA template (output_glca_binary_template)
#   - LCA-3-S template (output_lca3s_binary_template)
#
# Note: mixes methods (MSA for LCA, template for GLCA/LCA-3-S). Useful for
# seeing per-ligand metric behavior before the full MSA vs template comparison.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/analyze_available_results.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"
LABELS_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"
SCORING_DIR="${RESULTS_DIR}/scoring_partial"

# Conda activation
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi

mkdir -p "${RESULTS_DIR}" "${SCORING_DIR}"

# ══════════════════════════════════════════════════════════════════
# STEP 1: AGGREGATE AVAILABLE RESULTS
# ══════════════════════════════════════════════════════════════════

echo "============================================"
echo "Step 1: Aggregate available Boltz2 results"
echo "============================================"

# ── LCA (MSA, 2 dirs: binders + non-binders) ──
echo ""
echo "--- Lithocholic Acid (MSA) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/lca_msa_binary_output/output_tier1_binary" "${SCRATCH}/lca_msa_binary_output/output_tier4_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca_binary_results.csv"

# ── GLCA (template) ──
echo ""
echo "--- GlycoLithocholic Acid (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_glca_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_binary_template_results.csv"

# ── LCA-3-S (template) ──
echo ""
echo "--- Lithocholic Acid 3-Sulfate (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca3s_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_binary_template_results.csv"

# ══════════════════════════════════════════════════════════════════
# STEP 2: QUICK PER-LIGAND ANALYSIS
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 2: Quick per-ligand analysis"
echo "============================================"

# LCA MSA
LCA_RESULTS="${RESULTS_DIR}/boltz_lca_binary_results.csv"
LCA_LABELS="${LABELS_DIR}/boltz_lca_binary.csv"
if [ -f "${LCA_RESULTS}" ] && [ -f "${LCA_LABELS}" ]; then
    echo ""
    echo "--- LCA (MSA) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${LCA_RESULTS}" --labels "${LCA_LABELS}" 2>/dev/null || true
fi

# GLCA template
GLCA_RESULTS="${RESULTS_DIR}/boltz_glca_binary_template_results.csv"
GLCA_LABELS="${LABELS_DIR}/boltz_glca_binary.csv"
if [ -f "${GLCA_RESULTS}" ] && [ -f "${GLCA_LABELS}" ]; then
    echo ""
    echo "--- GLCA (template) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${GLCA_RESULTS}" --labels "${GLCA_LABELS}" 2>/dev/null || true
fi

# LCA-3-S template
LCA3S_RESULTS="${RESULTS_DIR}/boltz_lca3s_binary_template_results.csv"
LCA3S_LABELS="${LABELS_DIR}/boltz_lca3s_binary.csv"
if [ -f "${LCA3S_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    echo ""
    echo "--- LCA-3-S (template) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${LCA3S_RESULTS}" --labels "${LCA3S_LABELS}" 2>/dev/null || true
fi

# ══════════════════════════════════════════════════════════════════
# STEP 3: CROSS-LIGAND SCORING
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 3: Cross-ligand scoring (3 available results)"
echo "============================================"

DATA_ARGS=""

# LCA MSA
if [ -f "${LCA_RESULTS}" ] && [ -f "${LCA_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data LCA ${LCA_RESULTS} ${LCA_LABELS}"
fi

# GLCA template
if [ -f "${GLCA_RESULTS}" ] && [ -f "${GLCA_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data GLCA ${GLCA_RESULTS} ${GLCA_LABELS}"
fi

# LCA-3-S template
if [ -f "${LCA3S_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data LCA3S ${LCA3S_RESULTS} ${LCA3S_LABELS}"
fi

if [ -z "${DATA_ARGS}" ]; then
    echo "ERROR: No result/label CSV pairs found"
    exit 1
fi

# shellcheck disable=SC2086
python "${PROJECT_ROOT}/scripts/analyze_boltz_scoring.py" \
    ${DATA_ARGS} \
    --out-dir "${SCORING_DIR}"

# ══════════════════════════════════════════════════════════════════
# STEP 4: FIGURES
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 4: Generating figures"
echo "============================================"

# shellcheck disable=SC2086
python "${PROJECT_ROOT}/scripts/plot_boltz_scoring.py" \
    ${DATA_ARGS} \
    --out-dir "${SCORING_DIR}"

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "COMPLETE — Partial analysis (3 available results)"
echo "============================================"
echo ""
echo "Methods:  LCA=MSA, GLCA=template, LCA-3-S=template"
echo "Results:  ${SCORING_DIR}/"
echo ""
ls -la "${SCORING_DIR}/" 2>/dev/null || echo "(empty)"
