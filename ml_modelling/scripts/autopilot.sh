#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# autopilot.sh — Automated multi-tier pipeline runner
#
# Runs each tier's orchestrator, waits for SLURM jobs to complete, then
# re-runs until no more jobs are submitted. Moves to next tier automatically.
#
# Usage:
#   # Run interactively (e.g., in a tmux/screen session):
#   bash ml_modelling/scripts/autopilot.sh
#
#   # Or submit as a low-resource SLURM job:
#   sbatch --job-name=autopilot --time=24:00:00 --ntasks=1 --mem=4G \
#          --partition=amilan --account=ucb472_asc2 --qos=normal \
#          ml_modelling/scripts/autopilot.sh
#
# Re-entry safe: can be killed and restarted at any time.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.." || true
PROJECT_ROOT="$(pwd)"

POLL_INTERVAL=300  # seconds between queue checks (5 min)
MAX_WAIT=28800     # max seconds to wait for a single round of jobs (8 hours)

# ── Tier definitions ──────────────────────────────────────────────────────────
# Format: "csv_file|extra_args"
TIERS=(
    "tiers/tier1_strong_binders.csv|--batch-size 50 --relax-per-task 1 --workers 8"
    "tiers/tier2_win_ssm_graded.csv|--batch-size 50 --relax-per-task 1 --workers 8"
    "tiers/tier3_pnas_cutler.csv|--batch-size 50 --relax-per-task 10 --workers 8"
    "tiers/tier4_LCA_screen.csv|--batch-size 50 --relax-per-task 20 --workers 8"
    "tiers/tier5_artificial.csv|--batch-size 50 --relax-per-task 1 --workers 8"
)

# ── Helper functions ──────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

count_my_jobs() {
    # Count running + pending jobs (exclude this autopilot job itself)
    squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cv "autopilot" || echo 0
}

wait_for_jobs() {
    local waited=0
    local n_jobs

    n_jobs=$(count_my_jobs)
    if [[ "$n_jobs" -eq 0 ]]; then
        return 0
    fi

    log "Waiting for $n_jobs SLURM jobs to complete (polling every ${POLL_INTERVAL}s)..."

    while [[ "$waited" -lt "$MAX_WAIT" ]]; do
        sleep "$POLL_INTERVAL"
        waited=$((waited + POLL_INTERVAL))
        n_jobs=$(count_my_jobs)

        if [[ "$n_jobs" -eq 0 ]]; then
            log "All jobs completed after ${waited}s"
            return 0
        fi

        log "  Still waiting: $n_jobs jobs remaining (${waited}s elapsed)"
    done

    log "WARNING: Timed out after ${MAX_WAIT}s with $n_jobs jobs still running"
    log "  Continuing anyway — orchestrator will handle partial results"
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
    local max_rounds=10  # safety limit

    while [[ "$round" -le "$max_rounds" ]]; do
        log "Round $round for $tier_name"

        # Run the orchestrator
        # shellcheck disable=SC2086
        bash ml_modelling/scripts/run_tier.sh "$csv" $extra_args 2>&1 | \
            while IFS= read -r line; do echo "  $line"; done

        # Check if any jobs were submitted
        sleep 10  # give SLURM a moment to register new jobs
        local n_jobs
        n_jobs=$(count_my_jobs)

        if [[ "$n_jobs" -eq 0 ]]; then
            log "Tier $tier_name: No jobs submitted — tier complete!"
            break
        fi

        log "Tier $tier_name: $n_jobs jobs submitted, waiting..."
        wait_for_jobs

        round=$((round + 1))
    done

    if [[ "$round" -gt "$max_rounds" ]]; then
        log "WARNING: $tier_name hit max rounds ($max_rounds). Moving on."
    fi

    log ""
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "Autopilot starting — processing ${#TIERS[@]} tiers"
log "Poll interval: ${POLL_INTERVAL}s, Max wait per round: ${MAX_WAIT}s"
log ""

for tier_spec in "${TIERS[@]}"; do
    IFS='|' read -r csv extra_args <<< "$tier_spec"
    run_tier "$csv" "$extra_args"
done

log "════════════════════════════════════════════════════════════════"
log "ALL TIERS COMPLETE"
log "════════════════════════════════════════════════════════════════"
