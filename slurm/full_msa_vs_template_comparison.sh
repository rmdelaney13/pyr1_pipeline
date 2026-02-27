#!/bin/bash
# ============================================================================
# Full MSA vs template comparison across 3 bile acid ligands
# ============================================================================
# Prerequisites: ALL 6 Boltz2 runs must be complete:
#   MSA:      lca_msa_binary_output/output_tier1_binary + output_tier4_binary (LCA)
#             output_glca_binary (GLCA), output_lca3s_binary (LCA-3-S)
#   Template: output_lca_binary_template_v2 (LCA, corrected SMILES)
#             output_glca_binary_template (GLCA)
#             output_lca3s_binary_template (LCA-3-S)
#
# Produces:
#   - Per-ligand deep MSA vs template comparison (bootstrap AUC, paired
#     analysis, ensemble, variant-level deltas) with ~7 figures per ligand
#   - Pooled (all 3 ligands combined) MSA vs template comparison
#   - Cross-ligand scoring for MSA-only and template-only separately
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/full_msa_vs_template_comparison.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"
LABELS_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"

# Output subdirectories
SCORING_MSA="${RESULTS_DIR}/scoring_msa"
SCORING_TMPL="${RESULTS_DIR}/scoring_template"
COMPARE_DIR="${RESULTS_DIR}/msa_vs_template"

# Conda activation
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi
pip install matplotlib --quiet 2>/dev/null || true

mkdir -p "${RESULTS_DIR}" "${SCORING_MSA}" "${SCORING_TMPL}" "${COMPARE_DIR}"

# ══════════════════════════════════════════════════════════════════
# STEP 1: AGGREGATE ALL 6 RESULT SETS
# ══════════════════════════════════════════════════════════════════

echo "============================================"
echo "Step 1: Aggregate all Boltz2 results"
echo "============================================"

# ── MSA results ──

echo ""
echo "--- LCA (MSA, 2 dirs) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/lca_msa_binary_output/output_tier1_binary" "${SCRATCH}/lca_msa_binary_output/output_tier4_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca_binary_results.csv"

echo ""
echo "--- GLCA (MSA) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_glca_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_binary_results.csv"

echo ""
echo "--- LCA-3-S (MSA) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca3s_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_binary_results.csv"

# ── Template results ──

echo ""
echo "--- LCA (template v2, corrected SMILES) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca_binary_template_v2" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca_binary_template_results.csv"

echo ""
echo "--- GLCA (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_glca_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_binary_template_results.csv"

echo ""
echo "--- LCA-3-S (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca3s_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_binary_template_results.csv"

echo ""
echo "All 6 result CSVs aggregated."

# ══════════════════════════════════════════════════════════════════
# STEP 2: PER-LIGAND MSA VS TEMPLATE DEEP COMPARISON
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 2: Per-ligand MSA vs template deep comparison"
echo "============================================"

declare -A LIGAND_NAMES
LIGAND_NAMES[lca]="LCA"
LIGAND_NAMES[glca]="GLCA"
LIGAND_NAMES[lca3s]="LCA-3-S"

for LIGAND in lca glca lca3s; do
    MSA_CSV="${RESULTS_DIR}/boltz_${LIGAND}_binary_results.csv"
    TMPL_CSV="${RESULTS_DIR}/boltz_${LIGAND}_binary_template_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"
    LIGAND_OUT="${COMPARE_DIR}/${LIGAND}"

    if [ ! -f "${MSA_CSV}" ] || [ ! -f "${TMPL_CSV}" ] || [ ! -f "${LABELS}" ]; then
        echo "SKIP ${LIGAND}: missing input files"
        echo "  MSA:    ${MSA_CSV} $([ -f "${MSA_CSV}" ] && echo OK || echo MISSING)"
        echo "  TMPL:   ${TMPL_CSV} $([ -f "${TMPL_CSV}" ] && echo OK || echo MISSING)"
        echo "  LABELS: ${LABELS} $([ -f "${LABELS}" ] && echo OK || echo MISSING)"
        continue
    fi

    echo ""
    echo "--- ${LIGAND_NAMES[$LIGAND]} ---"
    mkdir -p "${LIGAND_OUT}"

    python "${PROJECT_ROOT}/scripts/deep_compare_msa_vs_template.py" \
        --msa-results "${MSA_CSV}" \
        --template-results "${TMPL_CSV}" \
        --labels "${LABELS}" \
        --ligand "${LIGAND_NAMES[$LIGAND]}" \
        --out-dir "${LIGAND_OUT}"

    # Generate additional figures
    PAIRED_CSV="${LIGAND_OUT}/paired_comparison.csv"
    if [ -f "${PAIRED_CSV}" ]; then
        python "${PROJECT_ROOT}/scripts/plot_msa_vs_template.py" \
            --csv "${PAIRED_CSV}" \
            --out-dir "${LIGAND_OUT}"
    fi
done

# ══════════════════════════════════════════════════════════════════
# STEP 2b: POOLED ANALYSIS (ALL 3 LIGANDS COMBINED)
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 2b: Pooled MSA vs template (all 3 ligands)"
echo "============================================"

POOLED_DIR="${COMPARE_DIR}/pooled"
mkdir -p "${POOLED_DIR}"

# Concatenate MSA results (header from first file only)
head -1 "${RESULTS_DIR}/boltz_lca_binary_results.csv" > "${POOLED_DIR}/msa_pooled.csv"
for LIGAND in lca glca lca3s; do
    CSV="${RESULTS_DIR}/boltz_${LIGAND}_binary_results.csv"
    if [ -f "${CSV}" ]; then
        tail -n+2 "${CSV}" >> "${POOLED_DIR}/msa_pooled.csv"
    fi
done

# Concatenate template results
head -1 "${RESULTS_DIR}/boltz_lca_binary_template_results.csv" > "${POOLED_DIR}/tmpl_pooled.csv"
for LIGAND in lca glca lca3s; do
    CSV="${RESULTS_DIR}/boltz_${LIGAND}_binary_template_results.csv"
    if [ -f "${CSV}" ]; then
        tail -n+2 "${CSV}" >> "${POOLED_DIR}/tmpl_pooled.csv"
    fi
done

# Concatenate labels
head -1 "${LABELS_DIR}/boltz_lca_binary.csv" > "${POOLED_DIR}/labels_pooled.csv"
for LIGAND in lca glca lca3s; do
    CSV="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"
    if [ -f "${CSV}" ]; then
        tail -n+2 "${CSV}" >> "${POOLED_DIR}/labels_pooled.csv"
    fi
done

MSA_POOLED="${POOLED_DIR}/msa_pooled.csv"
TMPL_POOLED="${POOLED_DIR}/tmpl_pooled.csv"
LABELS_POOLED="${POOLED_DIR}/labels_pooled.csv"

echo "Pooled MSA:      $(tail -n+2 "${MSA_POOLED}" | wc -l) variants"
echo "Pooled template: $(tail -n+2 "${TMPL_POOLED}" | wc -l) variants"
echo "Pooled labels:   $(tail -n+2 "${LABELS_POOLED}" | wc -l) labels"

python "${PROJECT_ROOT}/scripts/deep_compare_msa_vs_template.py" \
    --msa-results "${MSA_POOLED}" \
    --template-results "${TMPL_POOLED}" \
    --labels "${LABELS_POOLED}" \
    --ligand "POOLED" \
    --out-dir "${POOLED_DIR}"

PAIRED_CSV="${POOLED_DIR}/paired_comparison.csv"
if [ -f "${PAIRED_CSV}" ]; then
    python "${PROJECT_ROOT}/scripts/plot_msa_vs_template.py" \
        --csv "${PAIRED_CSV}" \
        --out-dir "${POOLED_DIR}"
fi

# ══════════════════════════════════════════════════════════════════
# STEP 3: CROSS-LIGAND SCORING (MSA-ONLY AND TEMPLATE-ONLY)
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "Step 3: Cross-ligand scoring"
echo "============================================"

# ── MSA scoring ──
echo ""
echo "--- MSA cross-ligand scoring ---"
MSA_DATA_ARGS=""
for LIGAND in lca glca lca3s; do
    RESULTS="${RESULTS_DIR}/boltz_${LIGAND}_binary_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"
    if [ -f "${RESULTS}" ] && [ -f "${LABELS}" ]; then
        MSA_DATA_ARGS="${MSA_DATA_ARGS} --data ${LIGAND^^} ${RESULTS} ${LABELS}"
    fi
done

if [ -n "${MSA_DATA_ARGS}" ]; then
    # shellcheck disable=SC2086
    python "${PROJECT_ROOT}/scripts/analyze_boltz_scoring.py" \
        ${MSA_DATA_ARGS} --out-dir "${SCORING_MSA}"
    # shellcheck disable=SC2086
    python "${PROJECT_ROOT}/scripts/plot_boltz_scoring.py" \
        ${MSA_DATA_ARGS} --out-dir "${SCORING_MSA}"
else
    echo "WARNING: No MSA results available"
fi

# ── Template scoring ──
echo ""
echo "--- Template cross-ligand scoring ---"
TMPL_DATA_ARGS=""
for LIGAND in lca glca lca3s; do
    RESULTS="${RESULTS_DIR}/boltz_${LIGAND}_binary_template_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"
    if [ -f "${RESULTS}" ] && [ -f "${LABELS}" ]; then
        TMPL_DATA_ARGS="${TMPL_DATA_ARGS} --data ${LIGAND^^} ${RESULTS} ${LABELS}"
    fi
done

if [ -n "${TMPL_DATA_ARGS}" ]; then
    # shellcheck disable=SC2086
    python "${PROJECT_ROOT}/scripts/analyze_boltz_scoring.py" \
        ${TMPL_DATA_ARGS} --out-dir "${SCORING_TMPL}"
    # shellcheck disable=SC2086
    python "${PROJECT_ROOT}/scripts/plot_boltz_scoring.py" \
        ${TMPL_DATA_ARGS} --out-dir "${SCORING_TMPL}"
else
    echo "WARNING: No template results available"
fi

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "COMPLETE — Full MSA vs Template Comparison"
echo "============================================"
echo ""
echo "Per-ligand deep comparison:"
for LIGAND in lca glca lca3s; do
    echo "  ${COMPARE_DIR}/${LIGAND}/"
done
echo "  ${COMPARE_DIR}/pooled/"
echo ""
echo "Cross-ligand scoring:"
echo "  MSA:      ${SCORING_MSA}/metric_ranking.csv"
echo "  Template: ${SCORING_TMPL}/metric_ranking.csv"
echo ""
echo "Key comparison files:"
for LIGAND in lca glca lca3s pooled; do
    CSV="${COMPARE_DIR}/${LIGAND}/paired_comparison.csv"
    [ -f "${CSV}" ] && echo "  ${CSV}" || echo "  ${CSV} (not found)"
done
