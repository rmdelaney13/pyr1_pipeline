#!/usr/bin/env bash
#SBATCH --job-name=autopilot
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --output=autopilot_%j.log
# ─────────────────────────────────────────────────────────────────────────────
# autopilot.sh — Self-resubmitting pipeline runner
#
# Runs the orchestrator for all tiers, then resubmits itself with
# --dependency=afterany so it wakes up only when all jobs finish.
# Zero wasted CPU — only runs for a few minutes each cycle.
#
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   sbatch ml_modelling/scripts/autopilot.sh
#
# To stop: cancel the autopilot job and any pending resubmission
#   scancel -u $USER -n autopilot
#
# Re-entry safe: can be cancelled and restarted at any time.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/../.." || true
PROJECT_ROOT="$(pwd)"

STATEFILE="$PROJECT_ROOT/ml_modelling/scripts/.autopilot_state"
MAX_ROUNDS=30  # safety limit across all resubmissions

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── Track round number via state file ─────────────────────────────────────────
round=1
if [[ -f "$STATEFILE" ]]; then
    round=$(cat "$STATEFILE")
    round=$((round + 1))
fi
echo "$round" > "$STATEFILE"

if [[ "$round" -gt "$MAX_ROUNDS" ]]; then
    log "Hit max rounds ($MAX_ROUNDS). Stopping autopilot."
    rm -f "$STATEFILE"
    exit 0
fi

log "════════════════════════════════════════════════════════════════"
log "AUTOPILOT ROUND $round / $MAX_ROUNDS"
log "════════════════════════════════════════════════════════════════"

# ── Tier definitions ──────────────────────────────────────────────────────────
declare -A TIER_ARGS
TIER_ARGS[tier1_strong_binders]="tiers/tier1_strong_binders.csv --batch-size 50 --relax-per-task 1 --workers 8"
TIER_ARGS[tier2_win_ssm_graded]="tiers/tier2_win_ssm_graded.csv --batch-size 50 --relax-per-task 1 --workers 8"
TIER_ARGS[tier3_pnas_cutler]="tiers/tier3_pnas_cutler.csv --batch-size 50 --relax-per-task 10 --workers 8"
TIER_ARGS[tier4_LCA_screen]="tiers/tier4_LCA_screen.csv --batch-size 50 --relax-per-task 20 --workers 8"
TIER_ARGS[tier5_artificial]="tiers/tier5_artificial.csv --batch-size 50 --relax-per-task 1 --workers 8"

TIER_ORDER=(tier1_strong_binders tier2_win_ssm_graded tier3_pnas_cutler tier4_LCA_screen tier5_artificial)

# ── Run orchestrator for each tier ────────────────────────────────────────────
for tier in "${TIER_ORDER[@]}"; do
    args="${TIER_ARGS[$tier]}"
    log "── Running: $tier ──"
    # shellcheck disable=SC2086
    bash ml_modelling/scripts/run_tier.sh $args 2>&1 | \
        while IFS= read -r line; do echo "    $line"; done
    log ""
done

# ── Check if any jobs were submitted ──────────────────────────────────────────
sleep 15  # let SLURM register new jobs
job_ids=$(squeue -u "$USER" -h -o "%A" 2>/dev/null | grep -v "^${SLURM_JOB_ID:-0}$" | sort -u | tr '\n' ':' | sed 's/:$//')

if [[ -z "$job_ids" ]]; then
    log "No SLURM jobs running — all tiers complete!"
    rm -f "$STATEFILE"
    exit 0
fi

n_jobs=$(echo "$job_ids" | tr ':' '\n' | wc -l)
log "$n_jobs unique jobs submitted. Resubmitting autopilot with dependency..."

# ── Resubmit self with dependency on all current jobs ─────────────────────────
new_job=$(sbatch \
    --dependency=afterany:${job_ids} \
    --job-name=autopilot \
    --time=00:30:00 \
    --ntasks=1 \
    --mem=4G \
    --partition=amilan \
    --account=ucb472_asc2 \
    --qos=normal \
    --output="${PROJECT_ROOT}/autopilot_%j.log" \
    --chdir="$PROJECT_ROOT" \
    "$PROJECT_ROOT/ml_modelling/scripts/autopilot.sh" 2>&1)

log "Resubmitted: $new_job"
log "  Dependencies: $job_ids"
log "  Next round: $((round + 1))"
log "════════════════════════════════════════════════════════════════"
