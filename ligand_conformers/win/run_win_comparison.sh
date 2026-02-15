#!/usr/bin/env bash
#
# run_win_comparison.sh — end-to-end Win 55,212-2 conformer comparison
#
# Generates conformers two ways (MMFF-only vs MMFF+OpenMM), then compares
# both against the 12 TREMD reference conformers.
#
# Run from the pyr1_pipeline repo root:
#   bash ligand_conformers/win/run_win_comparison.sh
#
# Requires: Python 3.9+, RDKit, numpy
# Optional: openmm + openmmforcefields (for the OpenMM run)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WIN_DIR="${REPO_ROOT}/ligand_conformers/win"
REF_SDF="${WIN_DIR}/win_12_conf.sdf"

# Win 55,212-2  (PubChem CID 5311501)
WIN_SMILES="CC1=C(C2=C3N1C(COC3=CC=C2)CN4CCOCC4)C(=O)C5=CC=CC6=CC=CC=C65"

# Output directories
MMFF_DIR="${WIN_DIR}/results_mmff"
OPENMM_DIR="${WIN_DIR}/results_openmm"
CMP_DIR="${WIN_DIR}/comparison"

echo "============================================================"
echo "  Win 55,212-2 Conformer Generation + TREMD Comparison"
echo "============================================================"
echo "SMILES: ${WIN_SMILES}"
echo "Reference: ${REF_SDF}"
echo ""

# ── Step 1: Generate conformers (MMFF-only) ──────────────────────────
echo "--- Step 1: MMFF-only conformer generation ---"
python -m ligand_conformers \
    --input "${WIN_SMILES}" \
    --input-type smiles \
    --outdir "${MMFF_DIR}" \
    --num-confs 500 \
    --seed 42 \
    --prune-rms 0.5 \
    --cluster-rmsd 1.25 \
    -k 12 \
    --energy-pre-filter-n 100 \
    --ligand-id win55212 \
    --nprocs 1
echo ""

# ── Step 2: Generate conformers (MMFF + OpenMM) ─────────────────────
# This step is skipped if OpenMM is not available.
echo "--- Step 2: MMFF + OpenMM conformer generation ---"
if python -c "import openmm; import openmmforcefields" 2>/dev/null; then
    python -m ligand_conformers \
        --input "${WIN_SMILES}" \
        --input-type smiles \
        --outdir "${OPENMM_DIR}" \
        --num-confs 500 \
        --seed 42 \
        --prune-rms 0.5 \
        --cluster-rmsd 1.25 \
        -k 12 \
        --energy-pre-filter-n 100 \
        --ligand-id win55212 \
        --nprocs 1 \
        --openmm-refine
    HAVE_OPENMM=1
else
    echo "  OpenMM not available — skipping OpenMM run."
    echo "  Install:  conda install -c conda-forge openmm openmmforcefields openff-toolkit"
    HAVE_OPENMM=0
fi
echo ""

# ── Step 3: Compare against TREMD ────────────────────────────────────
echo "--- Step 3: TREMD comparison ---"
mkdir -p "${CMP_DIR}"

if [ "${HAVE_OPENMM:-0}" = "1" ]; then
    # Compare both runs: raw pool and final selection for each
    python "${REPO_ROOT}/ligand_conformers/compare_tremd.py" \
        --ref "${REF_SDF}" \
        --gen \
            "${MMFF_DIR}/conformers_raw.sdf" \
            "${MMFF_DIR}/conformers_final.sdf" \
            "${OPENMM_DIR}/conformers_raw.sdf" \
            "${OPENMM_DIR}/conformers_final.sdf" \
        --label \
            "MMFF-all" \
            "MMFF-final-K" \
            "OpenMM-all" \
            "OpenMM-final-K" \
        --outdir "${CMP_DIR}"
else
    # MMFF only: compare raw pool and final selection
    python "${REPO_ROOT}/ligand_conformers/compare_tremd.py" \
        --ref "${REF_SDF}" \
        --gen \
            "${MMFF_DIR}/conformers_raw.sdf" \
            "${MMFF_DIR}/conformers_final.sdf" \
        --label \
            "MMFF-all" \
            "MMFF-final-K" \
        --outdir "${CMP_DIR}"
fi

echo ""
echo "============================================================"
echo "  Done. Results in:"
echo "    ${CMP_DIR}/tremd_comparison.csv"
echo "    ${CMP_DIR}/tremd_comparison_summary.json"
echo "============================================================"
