#!/bin/bash
# ============================================================================
# Test template-based binary Boltz2 predictions (no MSA)
# ============================================================================
# Run interactively on an Alpine login node. Creates template-based YAMLs
# for a few GLCA binders + non-binders and submits on atesting_a100.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/test_boltz_template.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
TEMPLATE="${PROJECT_ROOT}/structures/templates/Pyr1_LCA_mutant_template_converted.cif"
CONJUGATE_CSV="${PROJECT_ROOT}/ml_modelling/data/boltz_lca_conjugates/boltz_glca_binary.csv"

TEST_YAML_DIR="${SCRATCH}/test_template_inputs"
TEST_OUTPUT_DIR="${SCRATCH}/test_template_output"

# ── Step 0: Activate environment (skip if already active) ─────────────────
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "boltz_env" ]; then
    module load anaconda
    source activate boltz_env
fi

# ── Step 1: Verify prerequisites ──────────────────────────────────────────
echo "============================================"
echo "Verifying prerequisites"
echo "============================================"

if [ ! -f "${TEMPLATE}" ]; then
    echo "ERROR: Template CIF not found at ${TEMPLATE}"
    exit 1
fi
echo "Template: ${TEMPLATE}"

if [ ! -f "${CONJUGATE_CSV}" ]; then
    echo "ERROR: GLCA CSV not found at ${CONJUGATE_CSV}"
    exit 1
fi
echo "CSV: ${CONJUGATE_CSV}"

# ── Step 2: Create small test CSV (3 binders + 3 non-binders) ────────────
echo ""
echo "============================================"
echo "Creating test CSV (6 predictions)"
echo "============================================"

mkdir -p "${TEST_YAML_DIR}"

# Extract header + first 3 binders (label=1.0) + first 3 non-binders (label=0.0)
TEST_CSV="${TEST_YAML_DIR}/test_template.csv"
head -1 "${CONJUGATE_CSV}" > "${TEST_CSV}"
# Use awk to avoid SIGPIPE from grep|head with pipefail
awk -F',' '$6 == "1.0" && ++n <= 3' "${CONJUGATE_CSV}" >> "${TEST_CSV}"
awk -F',' '$6 == "0.0" && ++n <= 3' "${CONJUGATE_CSV}" >> "${TEST_CSV}"

echo "Test CSV: ${TEST_CSV}"
echo "Contents (pair_id, ligand_name, label):"
cut -d',' -f1,2,6 "${TEST_CSV}"

# ── Step 3: Generate template-based YAMLs ─────────────────────────────────
echo ""
echo "============================================"
echo "Generating template-based YAMLs"
echo "============================================"

python "${PROJECT_ROOT}/scripts/prepare_boltz_yamls.py" \
    "${TEST_CSV}" \
    --out-dir "${TEST_YAML_DIR}" \
    --mode binary \
    --template "${TEMPLATE}" \
    --affinity

echo ""
echo "Sample YAML:"
head -20 "$(head -1 "${TEST_YAML_DIR}/manifest.txt")"

# ── Step 4: Submit test job ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "Submitting test job"
echo "============================================"

MANIFEST="${TEST_YAML_DIR}/manifest.txt"
TOTAL=$(wc -l < "${MANIFEST}")

echo "Predictions: ${TOTAL}"
echo "Partition: atesting_a100"
echo "Output: ${TEST_OUTPUT_DIR}"

JOB=$(sbatch --partition=atesting_a100 --qos=testing --time=00:30:00 \
    --job-name=boltz_tmpl_test \
    "${PROJECT_ROOT}/slurm/submit_boltz.sh" \
    "${MANIFEST}" "${TEST_OUTPUT_DIR}" "${TOTAL}" 5 \
    | awk '{print $NF}')

echo "  Job: ${JOB}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Log:     tail -f ${SCRATCH}/boltz_${JOB}_0.out"
echo ""
echo "After completion, check:"
echo "  ls ${TEST_OUTPUT_DIR}/boltz_results_*/predictions/*/"
echo "  cat ${TEST_OUTPUT_DIR}/boltz_results_*/predictions/*/affinity_*.json"
