#!/bin/bash
# ============================================================================
# Z-score guided expansion: seed each round from filtered results
# ============================================================================
#
# Instead of selecting top designs by binary_total_score (default pipeline),
# this script seeds each round from the Z-score filtered output. After each
# Boltz round completes, it re-runs the full scoring+filtering pipeline to
# generate updated Z-score rankings, then seeds the next round.
#
# Usage (run in screen/tmux on login node):
#   cd /projects/ryde3462/software/pyr1_pipeline
#   screen -S zscore_exp
#   bash slurm/run_zscore_expansion.sh 5 7        # rounds 5-7 for all ligands
#   bash slurm/run_zscore_expansion.sh 5 7 ca     # rounds 5-7 for CA only
#
# Prerequisites:
#   - Existing filtered output in expansion/ligandmpnn/filtered/
#   - boltz_env conda environment active
#   - Previous rounds (0-4) already completed
#
# ============================================================================

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash slurm/run_zscore_expansion.sh <start_round> <end_round> [ligand]"
    echo "  Example: bash slurm/run_zscore_expansion.sh 5 7"
    echo "  Example: bash slurm/run_zscore_expansion.sh 5 7 ca"
    exit 1
fi

START_ROUND="$1"
END_ROUND="$2"
SINGLE_LIGAND="${3:-}"

PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
EXPANSION_ROOT="/scratch/alpine/ryde3462/expansion/ligandmpnn"
REF_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
FILTERED_DIR="${EXPANSION_ROOT}/filtered"
METHOD="ligandmpnn"

# Ligands to process
if [ -n "$SINGLE_LIGAND" ]; then
    LIGANDS=("${SINGLE_LIGAND,,}")
else
    LIGANDS=("ca" "cdca" "dca")
fi

POLL_INTERVAL=60  # seconds

# ── Helpers ─────────────────────────────────────────────────────────────────

wait_for_jobs() {
    local ligand="$1"
    local round="$2"
    while true; do
        local mpnn_count=$(squeue -u "$USER" -h -o "%.50j" 2>/dev/null | grep -c "mpnn_${ligand}_r${round}" || true)
        local boltz_count=$(squeue -u "$USER" -h -o "%.50j" 2>/dev/null | grep -c "boltz_exp_${ligand}_r${round}" || true)
        local total=$((mpnn_count + boltz_count))
        if [ "$total" -eq 0 ]; then
            return 0
        fi
        echo "  $(date +%H:%M:%S) Waiting: ${mpnn_count} MPNN + ${boltz_count} Boltz jobs for ${ligand^^} r${round}..."
        sleep "$POLL_INTERVAL"
    done
}

rescore_all() {
    # Delete stale boltz_scored.csv to force re-scoring with all rounds
    for lig in "${LIGANDS[@]}"; do
        local scored="${EXPANSION_ROOT}/${lig}/boltz_scored.csv"
        if [ -f "$scored" ]; then
            echo "  Deleting stale ${scored}"
            rm -f "$scored"
        fi
    done

    # Re-generate boltz_scored.csv from all round directories
    echo "  Re-scoring Boltz predictions..."
    for lig in "${LIGANDS[@]}"; do
        local lig_dir="${EXPANSION_ROOT}/${lig}"
        local scored="${lig_dir}/boltz_scored.csv"
        local boltz_dirs=()
        for rd in "${lig_dir}"/round_*/boltz_output; do
            [ -d "$rd" ] && boltz_dirs+=("$rd")
        done
        if [ ${#boltz_dirs[@]} -gt 0 ]; then
            python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
                --binary-dir "${boltz_dirs[@]}" \
                --ref-pdb "${REF_PDB}" \
                --out "${scored}"
            echo "  ${lig^^}: scored ${#boltz_dirs[@]} round dirs → ${scored}"
        else
            echo "  ${lig^^}: no boltz_output dirs found, skipping"
        fi
    done

    # Re-run the full filtering pipeline
    echo "  Re-running filtering pipeline..."
    python "${PROJECT_ROOT}/scripts/filter_expansion_designs.py" \
        --expansion-root "${EXPANSION_ROOT}" \
        --ligands "${LIGANDS[@]}" \
        --gate-plddt 0.65 \
        --gate-hbond 4.5 \
        --gate-max-unsatisfied-oh 1 \
        --gate-coo-satisfied \
        --gate-min-r116-dist 5.0 \
        --gate-latch-rmsd 1.25 \
        --top-n 100 \
        --extract-sequences \
        --ref-pdb "${REF_PDB}" \
        --exclude-mutations 117F
}

# ── Main loop ───────────────────────────────────────────────────────────────

echo "============================================"
echo "Z-score Guided Expansion"
echo "============================================"
echo "Ligands:   ${LIGANDS[*]}"
echo "Rounds:    ${START_ROUND} - ${END_ROUND}"
echo "Started:   $(date)"
echo ""

# Activate environment
module load anaconda 2>/dev/null || true
source activate boltz_env 2>/dev/null || true

for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Round ${ROUND}: Z-score guided expansion"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # ── Step 1: Score all rounds + filter with Z-score gates ──
    echo "Scoring all Boltz predictions and filtering..."
    rescore_all

    # ── Step 3: Seed each ligand's round from filtered results ──
    echo ""
    echo "Seeding round ${ROUND} from filtered results..."
    for LIG in "${LIGANDS[@]}"; do
        FILTERED_CSV="${FILTERED_DIR}/top100_${LIG}.csv"
        if [ ! -f "$FILTERED_CSV" ]; then
            echo "  WARNING: ${FILTERED_CSV} not found, skipping ${LIG^^}"
            continue
        fi

        ROUND_DIR="${EXPANSION_ROOT}/${LIG}/round_${ROUND}"
        if [ -f "${ROUND_DIR}/cumulative_scores.csv" ]; then
            echo "  ${LIG^^}: round ${ROUND} already complete, skipping"
            continue
        fi

        python "${PROJECT_ROOT}/scripts/seed_expansion_from_filtered.py" \
            --filtered-csv "${FILTERED_CSV}" \
            --expansion-root "${EXPANSION_ROOT}" \
            --ligand "${LIG}" \
            --round "${ROUND}"
    done

    # ── Step 4: Run Phase B+C for each ligand via run_expansion.sh ──
    # Phase A is already done (selected_pdbs/ seeded above).
    # run_expansion.sh will detect selected_pdbs/ and skip to MPNN submission.
    echo ""
    echo "Submitting MPNN jobs..."
    for LIG in "${LIGANDS[@]}"; do
        ROUND_DIR="${EXPANSION_ROOT}/${LIG}/round_${ROUND}"
        if [ -f "${ROUND_DIR}/cumulative_scores.csv" ]; then
            continue
        fi
        if [ ! -d "${ROUND_DIR}/selected_pdbs" ]; then
            echo "  ${LIG^^}: no selected_pdbs, skipping"
            continue
        fi
        echo ""
        echo "--- ${LIG^^} Phase A (MPNN submission) ---"
        bash "${PROJECT_ROOT}/slurm/run_expansion.sh" "${LIG}" "${ROUND}" "${METHOD}" || true
    done

    # ── Step 5: Wait for MPNN jobs to finish ──
    echo ""
    echo "Waiting for MPNN jobs..."
    for LIG in "${LIGANDS[@]}"; do
        wait_for_jobs "${LIG}" "${ROUND}"
    done
    echo "All MPNN jobs complete."

    # ── Step 6: Phase B (MPNN → CSV → YAML → submit Boltz) ──
    echo ""
    echo "Running Phase B (MPNN to Boltz)..."
    for LIG in "${LIGANDS[@]}"; do
        ROUND_DIR="${EXPANSION_ROOT}/${LIG}/round_${ROUND}"
        if [ -f "${ROUND_DIR}/cumulative_scores.csv" ]; then
            continue
        fi
        echo ""
        echo "--- ${LIG^^} Phase B ---"
        bash "${PROJECT_ROOT}/slurm/run_expansion.sh" "${LIG}" "${ROUND}" "${METHOD}" || true
    done

    # ── Step 7: Wait for Boltz jobs to finish ──
    echo ""
    echo "Waiting for Boltz jobs..."
    for LIG in "${LIGANDS[@]}"; do
        wait_for_jobs "${LIG}" "${ROUND}"
    done
    echo "All Boltz jobs complete."

    # ── Step 8: Phase C (score + merge) ──
    echo ""
    echo "Running Phase C (score + merge)..."
    for LIG in "${LIGANDS[@]}"; do
        ROUND_DIR="${EXPANSION_ROOT}/${LIG}/round_${ROUND}"
        if [ -f "${ROUND_DIR}/cumulative_scores.csv" ]; then
            continue
        fi
        echo ""
        echo "--- ${LIG^^} Phase C ---"
        bash "${PROJECT_ROOT}/slurm/run_expansion.sh" "${LIG}" "${ROUND}" "${METHOD}" || true
    done

    # ── Step 9: Delete stale scored CSVs for next round's re-filtering ──
    for LIG in "${LIGANDS[@]}"; do
        rm -f "${EXPANSION_ROOT}/${LIG}/boltz_scored.csv"
    done

    echo ""
    echo "Round ${ROUND} complete at $(date)"
done

# ── Final scoring with all rounds ──
echo ""
echo "============================================"
echo "Final scoring with all rounds"
echo "============================================"

# Score all rounds
for LIG in "${LIGANDS[@]}"; do
    BOLTZ_DIRS=()
    for rd in "${EXPANSION_ROOT}/${LIG}"/round_*/boltz_output; do
        [ -d "$rd" ] && BOLTZ_DIRS+=("$rd")
    done
    if [ ${#BOLTZ_DIRS[@]} -gt 0 ]; then
        python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
            --binary-dir "${BOLTZ_DIRS[@]}" \
            --ref-pdb "${REF_PDB}" \
            --out "${EXPANSION_ROOT}/${LIG}/boltz_scored.csv"
    fi
done

rescore_all

echo ""
echo "============================================"
echo "ALL DONE: Rounds ${START_ROUND}-${END_ROUND}"
echo "Finished: $(date)"
echo "============================================"
echo ""
echo "Outputs in: ${FILTERED_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review top designs per ligand"
echo "  2. Cluster for Twist ordering"
