#!/bin/bash
# ============================================================================
# Score all expansion Boltz2 predictions for bile acid designs
# ============================================================================
#
# Run this interactively on an Alpine login node (NOT as a SLURM job).
# It will:
#   1. Find all boltz_output directories across rounds for each ligand
#   2. Run analyze_boltz_output.py to extract confidence + geometry metrics
#   3. Run filter_expansion_designs.py to apply relaxed Strategy H gates
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   bash slurm/score_expansion_boltz.sh
#
# Prerequisites:
#   - boltz_env conda environment (has Biopython)
#   - Completed Boltz2 predictions in expansion/ligandmpnn/*/round_*/boltz_output/
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
EXPANSION_ROOT="/scratch/alpine/ryde3462/expansion/ligandmpnn"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"

# Ligands to process (skip UDCA — wrong SMILES)
LIGANDS=("ca" "cdca" "dca")

# Strategy H relaxed gates
GATE_PLDDT=0.65
GATE_HBOND=4.5
GATE_LATCH_RMSD=1.25
TOP_N=100

# ── Step 0: Activate environment ──────────────────────────────────────────

echo "============================================"
echo "Step 0: Activating environment"
echo "============================================"

module load anaconda
source activate boltz_env

# ── Step 1: Score Boltz2 predictions per ligand ───────────────────────────

echo ""
echo "============================================"
echo "Step 1: Score Boltz2 predictions per ligand"
echo "============================================"

for LIG in "${LIGANDS[@]}"; do
    LIG_DIR="${EXPANSION_ROOT}/${LIG}"

    if [ ! -d "${LIG_DIR}" ]; then
        echo "WARNING: ${LIG_DIR} not found, skipping ${LIG^^}"
        continue
    fi

    # Find all boltz_output directories across rounds
    BOLTZ_DIRS=()
    for round_dir in "${LIG_DIR}"/round_*/boltz_output; do
        if [ -d "$round_dir" ]; then
            BOLTZ_DIRS+=("$round_dir")
        fi
    done

    if [ ${#BOLTZ_DIRS[@]} -eq 0 ]; then
        echo "No boltz_output directories found for ${LIG^^}, skipping"
        continue
    fi

    OUT_CSV="${LIG_DIR}/boltz_scored.csv"

    # Skip if already scored (delete boltz_scored.csv to re-run)
    if [ -f "${OUT_CSV}" ]; then
        echo "${LIG^^}: ${OUT_CSV} already exists ($(wc -l < "${OUT_CSV}") lines). Delete to re-run."
        continue
    fi

    echo ""
    echo "--- ${LIG^^}: ${#BOLTZ_DIRS[@]} round directories ---"
    echo "  Dirs: ${BOLTZ_DIRS[*]}"

    python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
        --binary-dir "${BOLTZ_DIRS[@]}" \
        --ref-pdb "${REF_PDB}" \
        --out "${OUT_CSV}"

    echo "${LIG^^}: wrote ${OUT_CSV}"
done

# ── Step 2: Filter with relaxed Strategy H ────────────────────────────────

echo ""
echo "============================================"
echo "Step 2: Filter with relaxed Strategy H"
echo "============================================"
echo "  Gates: pLDDT_ligand >= ${GATE_PLDDT}, H-bond <= ${GATE_HBOND} Å"
echo "  Gates: max 1 unsatisfied OH, COO satisfied, no R116 salt bridge, latch RMSD <= ${GATE_LATCH_RMSD} Å"
echo "  Rank by: composite Z-score (OH_sat + ligand_pLDDT + pocket_pLDDT - hbond_dist)"
echo "  Top N: ${TOP_N}"
echo ""

python "${PROJECT_ROOT}/scripts/filter_expansion_designs.py" \
    --expansion-root "${EXPANSION_ROOT}" \
    --ligands "${LIGANDS[@]}" \
    --gate-plddt ${GATE_PLDDT} \
    --gate-hbond ${GATE_HBOND} \
    --gate-max-unsatisfied-oh 1 \
    --gate-coo-satisfied \
    --gate-min-r116-dist 5.0 \
    --gate-latch-rmsd ${GATE_LATCH_RMSD} \
    --top-n ${TOP_N} \
    --extract-sequences \
    --ref-pdb "${REF_PDB}" \
    --exclude-mutations 117F

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo ""
echo "Outputs in: ${EXPANSION_ROOT}/filtered/"
echo ""
echo "Next steps:"
echo "  1. Review top designs per ligand (top100_ca.csv, etc.)"
echo "  2. Check R116 flip rates — high flip rate may inflate confidence"
echo "  3. Use combined FASTA for Twist ordering"
