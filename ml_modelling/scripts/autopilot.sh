#!/usr/bin/env bash
#SBATCH --job-name=autopilot
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --output=autopilot_%j.log
# ─────────────────────────────────────────────────────────────────────────────
# autopilot.sh — Automated multi-tier pipeline runner (1-core polling)
#
# Runs each tier's orchestrator, waits for SLURM jobs to complete, then
# re-runs until no more jobs are submitted. Moves to next tier automatically.
# Uses 1 core for up to 24 hours — safe to disconnect after submitting.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   sbatch ml_modelling/scripts/autopilot.sh
#
# To monitor:  tail -f autopilot_*.log
# To stop:     scancel -u $USER -n autopilot
#
# Re-entry safe: can be cancelled and restarted at any time.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.." || true

POLL_INTERVAL=300  # seconds between queue checks (5 min)
MAX_WAIT=28800     # max wait per round of jobs (8 hours)

# ── Tier definitions ──────────────────────────────────────────────────────────
# Edit these to skip tiers or change args
TIERS=(
    "tiers/tier1_strong_binders.csv|--batch-size 50 --relax-per-task 20 --workers 8"
    "tiers/tier2_win_ssm_graded.csv|--batch-size 50 --relax-per-task 20 --workers 8"
    "tiers/tier3_pnas_cutler.csv|--batch-size 50 --docking-repeats 50 --relax-per-task 20 --workers 8"
    "tiers/tier4_LCA_screen.csv|--batch-size 50 --docking-repeats 50 --relax-per-task 20 --workers 8"
    "tiers/tier5_artificial.csv|--batch-size 50 --docking-repeats 50 --relax-per-task 20 --workers 8"
)

# ── Helper functions ──────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_my_jobs() {
    # Count running + pending jobs (exclude this autopilot job)
    local n
    n=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cv "autopilot" || true)
    echo "${n:-0}"
}

wait_for_jobs() {
    local waited=0
    local n_jobs

    n_jobs=$(count_my_jobs)
    if [[ "$n_jobs" -eq 0 ]]; then
        return 0
    fi

    log "  Waiting for $n_jobs jobs (polling every ${POLL_INTERVAL}s, max ${MAX_WAIT}s)..."

    while [[ "$waited" -lt "$MAX_WAIT" ]]; do
        sleep "$POLL_INTERVAL"
        waited=$((waited + POLL_INTERVAL))
        n_jobs=$(count_my_jobs)

        if [[ "$n_jobs" -eq 0 ]]; then
            log "  All jobs completed (${waited}s elapsed)"
            return 0
        fi

        # Compact status every poll
        log "  $n_jobs jobs remaining ($(( waited / 60 ))m elapsed)"
    done

    log "  WARNING: Timed out after ${MAX_WAIT}s with $n_jobs jobs still running"
    log "  Continuing — orchestrator will handle partial results"
    return 0
}

run_tier() {
    local csv="$1"
    local extra_args="$2"
    local tier_name
    tier_name=$(basename "$csv" .csv)

    log "════════════════════════════════════════════════════════════════"
    log "TIER: $tier_name"
    log "════════════════════════════════════════════════════════════════"

    local round=1
    local max_rounds=10

    while [[ "$round" -le "$max_rounds" ]]; do
        log "Round $round for $tier_name"

        # Run the orchestrator
        # shellcheck disable=SC2086
        bash ml_modelling/scripts/run_tier.sh "$csv" $extra_args 2>&1 | \
            while IFS= read -r line; do echo "    $line"; done

        # Check if any jobs were submitted
        sleep 15
        local n_jobs
        n_jobs=$(count_my_jobs)

        if [[ "$n_jobs" -eq 0 ]]; then
            log "✓ $tier_name complete (no jobs submitted)"
            break
        fi

        log "$tier_name: $n_jobs jobs submitted, waiting for completion..."
        wait_for_jobs

        round=$((round + 1))
    done

    if [[ "$round" -gt "$max_rounds" ]]; then
        log "WARNING: $tier_name hit max rounds ($max_rounds). Moving on."
    fi

    log ""
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "════════════════════════════════════════════════════════════════"
log "AUTOPILOT STARTED — ${#TIERS[@]} tiers to process"
log "════════════════════════════════════════════════════════════════"
log ""

for tier_spec in "${TIERS[@]}"; do
    IFS='|' read -r csv extra_args <<< "$tier_spec"
    run_tier "$csv" "$extra_args"
done

log "════════════════════════════════════════════════════════════════"
log "ALL TIERS COMPLETE"
log "════════════════════════════════════════════════════════════════"
