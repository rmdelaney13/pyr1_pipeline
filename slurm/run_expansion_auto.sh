#!/bin/bash
# ============================================================================
# Automated multi-round neural expansion
# ============================================================================
#
# Runs expansion rounds automatically by polling squeue between phases.
# Run this in a screen/tmux session on a login node.
#
# Usage:
#   screen -S expansion
#   bash slurm/run_expansion_auto.sh ca 1 4       # CA rounds 1-4
#   bash slurm/run_expansion_auto.sh cdca 1 3     # CDCA rounds 1-3
#
# Or all 4 ligands in detached screens:
#   screen -S exp_ca   -dm bash slurm/run_expansion_auto.sh ca   1 4
#   screen -S exp_cdca -dm bash slurm/run_expansion_auto.sh cdca 1 4
#   screen -S exp_udca -dm bash slurm/run_expansion_auto.sh udca 1 4
#   screen -S exp_dca  -dm bash slurm/run_expansion_auto.sh dca  1 4
#
# Prerequisites:
#   - Round 0 already scored (bash slurm/run_expansion.sh <lig> 0)
#
# ============================================================================

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash slurm/run_expansion_auto.sh <ligand> <start_round> <end_round>"
    echo "  Example: bash slurm/run_expansion_auto.sh ca 1 4"
    exit 1
fi

LIGAND="${1,,}"
START_ROUND="$2"
END_ROUND="$3"
POLL_INTERVAL=60  # seconds between squeue checks
SCRATCH="/scratch/alpine/ryde3462"

# ── Helper: wait for ALL expansion jobs for this ligand to finish ────────────
wait_for_ligand_jobs() {
    local label="${1:-jobs}"
    while true; do
        # Check for any MPNN or Boltz expansion jobs for this ligand
        local mpnn_count=$(squeue -u "$USER" -h 2>/dev/null | grep -c "mpnn_${LIGAND}" || true)
        local boltz_count=$(squeue -u "$USER" -h 2>/dev/null | grep -c "boltz_exp_${LIGAND}" || true)
        local total=$((mpnn_count + boltz_count))

        if [ "$total" -eq 0 ]; then
            return 0
        fi
        echo "  $(date +%H:%M:%S) - Waiting for ${label}: ${mpnn_count} MPNN + ${boltz_count} Boltz jobs for ${LIGAND^^}..."
        sleep "$POLL_INTERVAL"
    done
}

# ── Helper: check if round is complete ───────────────────────────────────────
round_complete() {
    local round="$1"
    local cumulative="${SCRATCH}/expansion_${LIGAND}/round_${round}/cumulative_scores.csv"
    [ -f "$cumulative" ]
}

# ── Main loop ────────────────────────────────────────────────────────────────

echo "============================================"
echo "Auto-expansion: ${LIGAND^^} rounds ${START_ROUND}-${END_ROUND}"
echo "============================================"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Started: $(date)"
echo ""

for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  ${LIGAND^^} — Round ${ROUND}                        "
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Skip if round already complete
    if round_complete "$ROUND"; then
        echo "Round ${ROUND} already complete, skipping."
        continue
    fi

    # Run up to 3 phases per round (A, B, C)
    # Each call to run_expansion.sh auto-detects which phase to run.
    # Between phases, wait for all SLURM jobs for this ligand to finish.
    for PHASE_NUM in 1 2 3; do
        # Check if round became complete (e.g., Phase C just ran)
        if round_complete "$ROUND"; then
            break
        fi

        # Wait for all running jobs for this ligand before next phase
        wait_for_ligand_jobs "phase ${PHASE_NUM} prerequisites"

        echo ""
        echo "── Phase call ${PHASE_NUM}/3 ──"
        # Run expansion script; capture output but also display it
        OUTPUT=$(bash slurm/run_expansion.sh "$LIGAND" "$ROUND" 2>&1) || true
        echo "$OUTPUT"

        # Check if the phase errored with BLOCKED (shouldn't happen after wait,
        # but handle gracefully)
        if echo "$OUTPUT" | grep -q "BLOCKED"; then
            echo "Unexpected BLOCKED state after waiting. Retrying in ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
            continue
        fi
    done

    if round_complete "$ROUND"; then
        echo ""
        echo "Round ${ROUND} complete at $(date)"
    else
        echo ""
        echo "WARNING: Round ${ROUND} did not complete after 3 phase calls."
        echo "Check logs and job status. Stopping."
        exit 1
    fi
done

echo ""
echo "============================================"
echo "ALL DONE: ${LIGAND^^} rounds ${START_ROUND}-${END_ROUND}"
echo "Finished: $(date)"
echo "============================================"
