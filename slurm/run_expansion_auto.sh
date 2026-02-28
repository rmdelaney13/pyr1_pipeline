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
# Or all 4 ligands (in separate screen windows):
#   screen -S exp_ca   && bash slurm/run_expansion_auto.sh ca   1 4
#   screen -S exp_cdca && bash slurm/run_expansion_auto.sh cdca 1 4
#   screen -S exp_udca && bash slurm/run_expansion_auto.sh udca 1 4
#   screen -S exp_dca  && bash slurm/run_expansion_auto.sh dca  1 4
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

LIGAND="$1"
START_ROUND="$2"
END_ROUND="$3"
POLL_INTERVAL=60  # seconds between squeue checks

# ── Helper: extract SLURM job ID from run_expansion.sh output ────────────────
extract_job_id() {
    # Looks for "job NNNNNNN submitted" in the output
    echo "$1" | grep -oP 'job \K[0-9]+' | tail -1
}

# ── Helper: wait for a specific SLURM job to complete ────────────────────────
wait_for_job() {
    local job_id="$1"
    local label="$2"

    if [ -z "$job_id" ]; then
        echo "WARNING: No job ID to wait for"
        return 0
    fi

    echo ""
    echo "Waiting for ${label} (job ${job_id})..."
    while true; do
        # Check if any tasks from this job are still in queue
        local remaining=$(squeue -j "$job_id" -h 2>/dev/null | wc -l)
        if [ "$remaining" -eq 0 ]; then
            echo "Job ${job_id} complete."
            return 0
        fi
        echo "  $(date +%H:%M:%S) - ${remaining} task(s) still running..."
        sleep "$POLL_INTERVAL"
    done
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

    # Phase A: Select + MPNN
    echo "── Phase A: Select + MPNN ──"
    OUTPUT_A=$(bash slurm/run_expansion.sh "$LIGAND" "$ROUND" 2>&1)
    echo "$OUTPUT_A"
    JOB_A=$(extract_job_id "$OUTPUT_A")
    wait_for_job "$JOB_A" "MPNN (Phase A)"

    # Phase B: Convert + Boltz
    echo ""
    echo "── Phase B: Convert + Boltz ──"
    OUTPUT_B=$(bash slurm/run_expansion.sh "$LIGAND" "$ROUND" 2>&1)
    echo "$OUTPUT_B"
    JOB_B=$(extract_job_id "$OUTPUT_B")
    wait_for_job "$JOB_B" "Boltz (Phase B)"

    # Phase C: Score + Merge
    echo ""
    echo "── Phase C: Score + Merge ──"
    OUTPUT_C=$(bash slurm/run_expansion.sh "$LIGAND" "$ROUND" 2>&1)
    echo "$OUTPUT_C"

    echo ""
    echo "Round ${ROUND} complete at $(date)"
done

echo ""
echo "============================================"
echo "ALL DONE: ${LIGAND^^} rounds ${START_ROUND}-${END_ROUND}"
echo "Finished: $(date)"
echo "============================================"
