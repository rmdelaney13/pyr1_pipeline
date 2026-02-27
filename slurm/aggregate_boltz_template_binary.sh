#!/bin/bash
# ============================================================================
# Aggregate template-based Boltz2 binary results for LCA, GLCA, LCA-3-S
# ============================================================================
# Run after all template SLURM jobs complete.
# Produces per-ligand CSVs with confidence, affinity, and geometry metrics.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/aggregate_boltz_template_binary.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Aggregating template-based binary results"
echo "============================================"

# ── LCA ──
echo ""
echo "--- Lithocholic Acid (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca_binary_template_results.csv"

# ── GlycoLCA ──
echo ""
echo "--- GlycoLithocholic Acid (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_glca_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_binary_template_results.csv"

# ── LCA-3-S ──
echo ""
echo "--- Lithocholic Acid 3-Sulfate (template) ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca3s_binary_template" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_binary_template_results.csv"

# ── Quick analysis ──
echo ""
echo "============================================"
echo "Quick statistical analysis"
echo "============================================"

LABELS_DIR="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates"

for LIGAND in lca glca lca3s; do
    CSV="${RESULTS_DIR}/boltz_${LIGAND}_binary_template_results.csv"
    LABELS="${LABELS_DIR}/boltz_${LIGAND}_binary.csv"
    if [ -f "$CSV" ]; then
        echo ""
        echo "--- ${LIGAND} (template) ---"
        if [ -f "$LABELS" ]; then
            python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" "$CSV" --labels "$LABELS" 2>/dev/null || true
        else
            python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" "$CSV" 2>/dev/null || true
        fi
    fi
done

echo ""
echo "============================================"
echo "Results written to: ${RESULTS_DIR}/"
echo "============================================"
ls -la "${RESULTS_DIR}/"*template*.csv 2>/dev/null || echo "(no template CSVs found)"
