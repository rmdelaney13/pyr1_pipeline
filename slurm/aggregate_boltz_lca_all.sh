#!/bin/bash
# ============================================================================
# Aggregate Boltz2 binary results for ALL three LCA ligands
# ============================================================================
# Run after all SLURM jobs complete. Produces three CSVs + quick analysis.
#
# Usage:
#   cd /projects/ryde3462/pyr1_pipeline
#   bash slurm/aggregate_boltz_lca_all.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
RESULTS_DIR="${PROJECT_ROOT}/ml_modelling/analysis/boltz_LCA"

module load anaconda
source activate boltz_env

mkdir -p "${RESULTS_DIR}"

echo "============================================"
echo "Aggregating Boltz2 binary results"
echo "============================================"

# ── LCA ──
echo ""
echo "--- Lithocholic Acid ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca_binary_results.csv"

# ── GlycoLCA ──
echo ""
echo "--- GlycoLithocholic Acid ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_glca_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_glca_binary_results.csv"

# ── LCA-3-S ──
echo ""
echo "--- Lithocholic Acid 3-Sulfate ---"
python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
    --binary-dir "${SCRATCH}/output_lca3s_binary" \
    --ref-pdb "${REF_PDB}" \
    --out "${RESULTS_DIR}/boltz_lca3s_binary_results.csv"

# ── Quick analysis on each ──
echo ""
echo "============================================"
echo "Quick statistical analysis"
echo "============================================"

for LIGAND in lca glca lca3s; do
    CSV="${RESULTS_DIR}/boltz_${LIGAND}_binary_results.csv"
    if [ -f "$CSV" ]; then
        echo ""
        echo "--- ${LIGAND} ---"
        python "${PROJECT_ROOT}/scripts/quick_boltz_analysis.py" "$CSV" 2>/dev/null || true
    fi
done

echo ""
echo "============================================"
echo "Results written to: ${RESULTS_DIR}/"
echo "============================================"
ls -la "${RESULTS_DIR}/"*.csv 2>/dev/null || echo "(no CSVs found)"
