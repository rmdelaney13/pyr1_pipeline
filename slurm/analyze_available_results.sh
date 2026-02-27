#!/bin/bash
# ============================================================================
# Intermediate analysis: available Boltz2 binary results
# ============================================================================
# Analyzes the 4 completed runs:
#   - LCA MSA (lca_msa_binary_output/output_tier1_binary + output_tier4_binary)
#   - GLCA template (output_glca_binary_template)
#   - LCA-3-S MSA (output_lca3s_binary)
#   - LCA-3-S template (output_lca3s_binary_template)
#
# Since both MSA and template are available for LCA-3-S, also runs a
# deep MSA vs template comparison for that ligand.
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
COMPARE_DIR="${RESULTS_DIR}/msa_vs_template"

# Conda activation
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi
pip install matplotlib --quiet 2>/dev/null || true

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

# ── GLCA (template only) ──
echo ""
echo "--- GlycoLithocholic Acid (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_glca_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_binary_template_results.csv"

# ── LCA-3-S (MSA) ──
echo ""
echo "--- Lithocholic Acid 3-Sulfate (MSA) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca3s_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_binary_results.csv"

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
GLCA_TMPL_RESULTS="${RESULTS_DIR}/boltz_glca_binary_template_results.csv"
GLCA_LABELS="${LABELS_DIR}/boltz_glca_binary.csv"
if [ -f "${GLCA_TMPL_RESULTS}" ] && [ -f "${GLCA_LABELS}" ]; then
    echo ""
    echo "--- GLCA (template) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${GLCA_TMPL_RESULTS}" --labels "${GLCA_LABELS}" 2>/dev/null || true
fi

# LCA-3-S MSA
LCA3S_MSA_RESULTS="${RESULTS_DIR}/boltz_lca3s_binary_results.csv"
LCA3S_LABELS="${LABELS_DIR}/boltz_lca3s_binary.csv"
if [ -f "${LCA3S_MSA_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    echo ""
    echo "--- LCA-3-S (MSA) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${LCA3S_MSA_RESULTS}" --labels "${LCA3S_LABELS}" 2>/dev/null || true
fi

# LCA-3-S template
LCA3S_TMPL_RESULTS="${RESULTS_DIR}/boltz_lca3s_binary_template_results.csv"
if [ -f "${LCA3S_TMPL_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    echo ""
    echo "--- LCA-3-S (template) ---"
    python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" \
        "${LCA3S_TMPL_RESULTS}" --labels "${LCA3S_LABELS}" 2>/dev/null || true
fi

# ══════════════════════════════════════════════════════════════════
# STEP 3: CROSS-LIGAND SCORING
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 3: Cross-ligand scoring (best available per ligand)"
echo "============================================"

DATA_ARGS=""

# LCA MSA (only method available)
if [ -f "${LCA_RESULTS}" ] && [ -f "${LCA_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data LCA_MSA ${LCA_RESULTS} ${LCA_LABELS}"
fi

# GLCA template (only method available)
if [ -f "${GLCA_TMPL_RESULTS}" ] && [ -f "${GLCA_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data GLCA_TMPL ${GLCA_TMPL_RESULTS} ${GLCA_LABELS}"
fi

# LCA-3-S MSA
if [ -f "${LCA3S_MSA_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data LCA3S_MSA ${LCA3S_MSA_RESULTS} ${LCA3S_LABELS}"
fi

# LCA-3-S template
if [ -f "${LCA3S_TMPL_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    DATA_ARGS="${DATA_ARGS} --data LCA3S_TMPL ${LCA3S_TMPL_RESULTS} ${LCA3S_LABELS}"
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
echo "Step 4: Generating scoring figures"
echo "============================================"

# shellcheck disable=SC2086
python "${PROJECT_ROOT}/scripts/plot_boltz_scoring.py" \
    ${DATA_ARGS} \
    --out-dir "${SCORING_DIR}"

# ══════════════════════════════════════════════════════════════════
# STEP 5: LCA-3-S DEEP MSA VS TEMPLATE COMPARISON
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 5: LCA-3-S deep MSA vs template comparison"
echo "============================================"

LCA3S_COMPARE="${COMPARE_DIR}/lca3s"
mkdir -p "${LCA3S_COMPARE}"

if [ -f "${LCA3S_MSA_RESULTS}" ] && [ -f "${LCA3S_TMPL_RESULTS}" ] && [ -f "${LCA3S_LABELS}" ]; then
    python "${PROJECT_ROOT}/scripts/deep_compare_msa_vs_template.py" \
        --msa-results "${LCA3S_MSA_RESULTS}" \
        --template-results "${LCA3S_TMPL_RESULTS}" \
        --labels "${LCA3S_LABELS}" \
        --ligand "LCA-3-S" \
        --out-dir "${LCA3S_COMPARE}"

    PAIRED_CSV="${LCA3S_COMPARE}/paired_comparison.csv"
    if [ -f "${PAIRED_CSV}" ]; then
        python "${PROJECT_ROOT}/scripts/plot_msa_vs_template.py" \
            --csv "${PAIRED_CSV}" \
            --out-dir "${LCA3S_COMPARE}" \
            --ligand "LCA-3-S"
    fi
else
    echo "SKIP: Missing MSA or template results for LCA-3-S"
fi

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "COMPLETE — Partial analysis (4 results + LCA-3-S comparison)"
echo "============================================"
echo ""
echo "Methods:  LCA=MSA, GLCA=template, LCA-3-S=MSA+template"
echo ""
echo "Scoring results:  ${SCORING_DIR}/"
ls -la "${SCORING_DIR}/"*.csv 2>/dev/null || echo "  (no CSVs)"
echo ""
echo "LCA-3-S MSA vs template:  ${LCA3S_COMPARE}/"
ls -la "${LCA3S_COMPARE}/"*.csv 2>/dev/null || echo "  (no CSVs)"
