#!/bin/bash
# ============================================================================
# Submit remaining Boltz2 binary runs:
#   1. LCA template (rerun with corrected SMILES)
#   2. LCA-3-S MSA (max_msa_seqs=32 already in submit_boltz.sh)
#   3. GLCA MSA    (max_msa_seqs=32 already in submit_boltz.sh)
#
# All YAMLs include affinity property -> Boltz predicts affinity automatically.
#
# Run interactively on Alpine login node:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/submit_remaining_boltz_runs.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
SCRATCH="/scratch/alpine/ryde3462/boltz_lca"
SUBMIT_SCRIPT="${PROJECT_ROOT}/slurm/submit_boltz.sh"

BATCH_SIZE=20
DIFFUSION_SAMPLES=5

# ── Input / output directories ───────────────────────────────────────────

# 1) LCA template (rerun — SMILES fixed)
LCA_TPL_YAML="${SCRATCH}/inputs_lca_template"
LCA_TPL_OUT="${SCRATCH}/output_lca_binary_template_v2"

# 2) LCA-3-S MSA
LCA3S_YAML="${SCRATCH}/inputs_lca3s_prod"
LCA3S_OUT="${SCRATCH}/output_lca3s_binary"

# 3) GLCA MSA
GLCA_YAML="${SCRATCH}/inputs_glca_prod"
GLCA_OUT="${SCRATCH}/output_glca_binary"

# ── Helper: ensure manifest exists, compute array range, submit ──────────
submit_job() {
    local LABEL="$1"
    local YAML_DIR="$2"
    local OUT_DIR="$3"
    local TIME="$4"

    local MANIFEST="${YAML_DIR}/manifest.txt"

    # If no manifest, generate one from *.yaml
    if [ ! -f "${MANIFEST}" ]; then
        echo "${LABEL}: No manifest found, generating from *.yaml..."
        ls "${YAML_DIR}"/*.yaml | sort > "${MANIFEST}"
    fi

    local TOTAL
    TOTAL=$(wc -l < "${MANIFEST}")
    local ARRAY_MAX=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE - 1 ))

    echo "${LABEL}: ${TOTAL} YAMLs -> array=0-${ARRAY_MAX} (batch=${BATCH_SIZE}, time=${TIME})"

    mkdir -p "${OUT_DIR}"

    local JOB
    JOB=$(sbatch --array=0-${ARRAY_MAX} \
        --time="${TIME}" \
        --job-name="boltz_${LABEL}" \
        "${SUBMIT_SCRIPT}" \
        "${MANIFEST}" "${OUT_DIR}" "${BATCH_SIZE}" "${DIFFUSION_SAMPLES}" \
        | awk '{print $NF}')

    echo "  -> Job ${JOB}"
    eval "${LABEL}_JOB=${JOB}"
    eval "${LABEL}_TOTAL=${TOTAL}"
}

# ── Submit all three ─────────────────────────────────────────────────────

echo "============================================"
echo "Submitting 3 remaining Boltz2 binary runs"
echo "============================================"
echo ""

submit_job "lca_tpl"  "${LCA_TPL_YAML}"  "${LCA_TPL_OUT}"  "01:40:00"
echo ""
submit_job "lca3s"    "${LCA3S_YAML}"     "${LCA3S_OUT}"    "01:40:00"
echo ""
submit_job "glca"     "${GLCA_YAML}"      "${GLCA_OUT}"     "01:40:00"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "ALL SUBMITTED"
echo "============================================"
echo ""
echo "  LCA template (v2):  ${lca_tpl_JOB}  (${lca_tpl_TOTAL} predictions) -> ${LCA_TPL_OUT}"
echo "  LCA-3-S MSA:        ${lca3s_JOB}  (${lca3s_TOTAL} predictions) -> ${LCA3S_OUT}"
echo "  GLCA MSA:           ${glca_JOB}  (${glca_TOTAL} predictions) -> ${GLCA_OUT}"
echo ""
echo "Monitor: squeue -u \$USER"
