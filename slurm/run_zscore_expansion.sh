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
    # Incrementally update boltz_scored.csv — only score new rounds
    echo "  Updating Boltz scores (incremental)..."
    for lig in "${LIGANDS[@]}"; do
        local lig_dir="${EXPANSION_ROOT}/${lig}"
        local scored="${lig_dir}/boltz_scored.csv"
        local marker="${lig_dir}/.last_scored_round"
        local last_scored=-1

        if [ -f "$marker" ] && [ -f "$scored" ]; then
            last_scored=$(cat "$marker")
        fi

        # Find round dirs with boltz_output that haven't been scored yet
        local new_dirs=()
        local max_round=$last_scored
        for rd in "${lig_dir}"/round_*/boltz_output; do
            [ -d "$rd" ] || continue
            local rn=$(basename "$(dirname "$rd")" | sed 's/round_//')
            if [ "$rn" -gt "$last_scored" ]; then
                new_dirs+=("$rd")
                [ "$rn" -gt "$max_round" ] && max_round="$rn"
            fi
        done

        if [ ${#new_dirs[@]} -eq 0 ]; then
            echo "  ${lig^^}: boltz_scored.csv up to date (through round ${last_scored})"
            continue
        fi

        echo "  ${lig^^}: scoring ${#new_dirs[@]} new round dir(s) (rounds ${last_scored}+1 to ${max_round})..."
        local tmp="${lig_dir}/boltz_scored_incremental.csv"
        python "${PROJECT_ROOT}/scripts/analyze_boltz_output.py" \
            --binary-dir "${new_dirs[@]}" \
            --ref-pdb "${REF_PDB}" \
            --out "${tmp}"

        if [ -f "$scored" ]; then
            # Append new rows (skip header)
            tail -n +2 "$tmp" >> "$scored"
            rm "$tmp"
        else
            mv "$tmp" "$scored"
        fi

        echo "$max_round" > "$marker"
        echo "  ${lig^^}: updated boltz_scored.csv (through round ${max_round})"
    done

    # Run the filtering pipeline
    echo "  Running filtering pipeline..."
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

    echo ""
    echo "Round ${ROUND} complete at $(date)"
done

# ── Final filtering with all rounds ──
echo ""
echo "============================================"
echo "Final filtering with all rounds"
echo "============================================"

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
