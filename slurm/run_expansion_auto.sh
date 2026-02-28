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
#   bash slurm/run_expansion_auto.sh ca 1 4                   # LigandMPNN
#   bash slurm/run_expansion_auto.sh ca 1 4 lasermpnn         # LASErMPNN
#
# Or all 4 ligands in detached screens:
#   screen -S exp_ca   -dm bash slurm/run_expansion_auto.sh ca   1 4
#   screen -S exp_cdca -dm bash slurm/run_expansion_auto.sh cdca 1 4
#   screen -S exp_udca -dm bash slurm/run_expansion_auto.sh udca 1 4
#   screen -S exp_dca  -dm bash slurm/run_expansion_auto.sh dca  1 4
#
# LASErMPNN for all 4:
#   screen -S laser_ca   -dm bash slurm/run_expansion_auto.sh ca   1 4 lasermpnn
#   screen -S laser_cdca -dm bash slurm/run_expansion_auto.sh cdca 1 4 lasermpnn
#   screen -S laser_udca -dm bash slurm/run_expansion_auto.sh udca 1 4 lasermpnn
#   screen -S laser_dca  -dm bash slurm/run_expansion_auto.sh dca  1 4 lasermpnn
#
# Prerequisites:
#   - Round 0 already scored (bash slurm/run_expansion.sh <lig> 0 [method])
#
# ============================================================================

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash slurm/run_expansion_auto.sh <ligand> <start_round> <end_round> [method]"
    echo "  Example: bash slurm/run_expansion_auto.sh ca 1 4"
    echo "  Example: bash slurm/run_expansion_auto.sh ca 1 4 lasermpnn"
    exit 1
fi

LIGAND="${1,,}"
START_ROUND="$2"
END_ROUND="$3"
METHOD="${4:-ligandmpnn}"
POLL_INTERVAL=60  # seconds between squeue checks
SCRATCH="/scratch/alpine/ryde3462"

# Method-specific settings
if [ "$METHOD" = "lasermpnn" ]; then
    DESIGN_JOB_PATTERN="laser_${LIGAND}"
    BOLTZ_JOB_PATTERN="boltz_lexp_${LIGAND}"
    MAX_PHASES=4   # A, A', B, C
else
    DESIGN_JOB_PATTERN="mpnn_${LIGAND}"
    BOLTZ_JOB_PATTERN="boltz_exp_${LIGAND}"
    MAX_PHASES=3   # A, B, C
fi

# ── Helper: wait for ALL expansion jobs for this ligand to finish ────────────
wait_for_ligand_jobs() {
    local label="${1:-jobs}"
    while true; do
        # Use --format with wide name field to avoid squeue truncating job names
        # (default format truncates NAME to ~8 chars, breaking grep matching)
        local design_count=$(squeue -u "$USER" -h -o "%.50j" 2>/dev/null | grep -c "${DESIGN_JOB_PATTERN}" || true)
        local boltz_count=$(squeue -u "$USER" -h -o "%.50j" 2>/dev/null | grep -c "${BOLTZ_JOB_PATTERN}" || true)
        local total=$((design_count + boltz_count))

        if [ "$total" -eq 0 ]; then
            return 0
        fi
        echo "  $(date +%H:%M:%S) - Waiting for ${label}: ${design_count} ${METHOD} + ${boltz_count} Boltz jobs for ${LIGAND^^}..."
        sleep "$POLL_INTERVAL"
    done
}

# ── Helper: check if round is complete ───────────────────────────────────────
round_complete() {
    local round="$1"
    local cumulative="${SCRATCH}/expansion/${METHOD}/${LIGAND}/round_${round}/cumulative_scores.csv"
    [ -f "$cumulative" ]
}

# ── Main loop ────────────────────────────────────────────────────────────────

echo "============================================"
echo "Auto-expansion: ${LIGAND^^} rounds ${START_ROUND}-${END_ROUND} (${METHOD})"
echo "============================================"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Phases per round: ${MAX_PHASES}"
echo "Started: $(date)"
echo ""

for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  ${LIGAND^^} ${METHOD} — Round ${ROUND}              "
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Skip if round already complete
    if round_complete "$ROUND"; then
        echo "Round ${ROUND} already complete, skipping."
        continue
    fi

    # Run up to MAX_PHASES phases per round
    # Each call to run_expansion.sh auto-detects which phase to run.
    # Between phases, wait for all SLURM jobs for this ligand to finish.
    for PHASE_NUM in $(seq 1 $MAX_PHASES); do
        # Check if round became complete (e.g., Phase C just ran)
        if round_complete "$ROUND"; then
            break
        fi

        # Wait for all running jobs for this ligand before next phase
        wait_for_ligand_jobs "phase ${PHASE_NUM} prerequisites"

        echo ""
        echo "── Phase call ${PHASE_NUM}/${MAX_PHASES} ──"
        # Run expansion script; capture output but also display it
        OUTPUT=$(bash slurm/run_expansion.sh "$LIGAND" "$ROUND" "$METHOD" 2>&1) || true
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
        echo "WARNING: Round ${ROUND} did not complete after ${MAX_PHASES} phase calls."
        echo "Check logs and job status. Stopping."
        exit 1
    fi
done

echo ""
echo "============================================"
echo "ALL DONE: ${LIGAND^^} ${METHOD} rounds ${START_ROUND}-${END_ROUND}"
echo "Finished: $(date)"
echo "============================================"
