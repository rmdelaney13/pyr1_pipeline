#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tier.sh — Run the ML dataset pipeline on a tier of pairs, with
#               automatic batch splitting and SLURM-aware re-entry.
#
# Usage:
#   ./run_tier.sh <tier_csv> [options]
#
# Examples:
#   # Run tier 1 (strong binders) with defaults
#   ./run_tier.sh tiers/tier1_strong_binders.csv
#
#   # Run tier 2 with custom batch size and docking params
#   ./run_tier.sh tiers/tier2_win_ssm_graded.csv --batch-size 30 --docking-arrays 5
#
#   # Dry run — show what would be submitted without submitting
#   ./run_tier.sh tiers/tier3_pnas_cutler.csv --dry-run
#
#   # Run a single pre-split batch directly
#   ./run_tier.sh tiers/tier4_LCA_screen_batch03.csv --no-split
#
# This script:
#   1. Splits the tier CSV into batches (unless --no-split)
#   2. For each batch, runs the orchestrator in SLURM mode
#   3. The orchestrator's re-entry pattern handles stage progression —
#      just re-run this script after SLURM jobs complete to advance.
#   4. Reports status of each batch.
#
# Re-entry: Safe to run repeatedly. Completed pairs are cached via
#           metadata.json and will be skipped automatically.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Project paths (absolute, cluster) ───────────────────────────────────────
PROJECT_ROOT="/projects/ryde3462/software/pyr1_pipeline"
ORCHESTRATOR="${PROJECT_ROOT}/ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py"
BUILD_PAIRS="${PROJECT_ROOT}/ml_modelling/scripts/build_master_pairs.py"
TEMPLATE_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_nolig_H2O.pdb"
REFERENCE_PDB="${PROJECT_ROOT}/docking/ligand_alignment/files_for_PYR1_docking/3QN1_H2O.pdb"
CACHE_BASE="/scratch/alpine/ryde3462/ml_dataset"
DATA_DIR="${PROJECT_ROOT}/ml_modelling/data"

# ── Defaults ────────────────────────────────────────────────────────────────
BATCH_SIZE=50
DOCKING_REPEATS=50
DOCKING_ARRAYS=10
WORKERS=4
RELAX_PER_TASK=4
DRY_RUN=false
NO_SPLIT=false
SKIP_AF3=false

# ── Parse arguments ─────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 <tier_csv> [options]"
    echo ""
    echo "Options:"
    echo "  --batch-size N       Pairs per batch (default: $BATCH_SIZE)"
    echo "  --docking-repeats N  Docking repeats per conformer (default: $DOCKING_REPEATS)"
    echo "  --docking-arrays N   SLURM array tasks for docking (default: $DOCKING_ARRAYS)"
    echo "  --workers N          Parallel workers for pair processing (default: $WORKERS)"
    echo "  --relax-per-task N   Structures per relax SLURM task (default: $RELAX_PER_TASK)"
    echo "  --skip-af3           Skip AF3 stage (useful for early tiers)"
    echo "  --no-split           Don't split; run the CSV as-is"
    echo "  --dry-run            Show what would run without executing"
    echo "  --cache-dir DIR      Override cache directory"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

TIER_CSV="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)    BATCH_SIZE="$2";       shift 2 ;;
        --docking-repeats) DOCKING_REPEATS="$2"; shift 2 ;;
        --docking-arrays) DOCKING_ARRAYS="$2";   shift 2 ;;
        --workers)       WORKERS="$2";           shift 2 ;;
        --relax-per-task) RELAX_PER_TASK="$2";  shift 2 ;;
        --skip-af3)      SKIP_AF3=true;          shift ;;
        --no-split)      NO_SPLIT=true;         shift ;;
        --dry-run)       DRY_RUN=true;          shift ;;
        --cache-dir)     CACHE_BASE="$2";       shift 2 ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

# Resolve tier CSV path
if [[ ! -f "$TIER_CSV" ]]; then
    # Try relative to data dir
    if [[ -f "${DATA_DIR}/${TIER_CSV}" ]]; then
        TIER_CSV="${DATA_DIR}/${TIER_CSV}"
    else
        echo "ERROR: Cannot find tier CSV: $TIER_CSV"
        exit 1
    fi
fi

TIER_NAME=$(basename "$TIER_CSV" .csv)
CACHE_DIR="${CACHE_BASE}/${TIER_NAME}"

echo "════════════════════════════════════════════════════════════════"
echo "  ML Dataset Pipeline — Tier Runner"
echo "════════════════════════════════════════════════════════════════"
echo "  Tier CSV:        $TIER_CSV"
echo "  Cache dir:       $CACHE_DIR"
echo "  Batch size:      $BATCH_SIZE"
echo "  Docking:         $DOCKING_REPEATS repeats × $DOCKING_ARRAYS arrays"
echo "  Workers:         $WORKERS"
echo "  Relax/task:      $RELAX_PER_TASK structures per SLURM task"
echo "  Skip AF3:        $SKIP_AF3"
echo "  Dry run:         $DRY_RUN"

TOTAL_PAIRS=$(tail -n +2 "$TIER_CSV" | wc -l)
echo "  Total pairs:     $TOTAL_PAIRS"

# ── Estimate SLURM job usage ───────────────────────────────────────────────
RELAX_TASKS_PER_PAIR=$(( (20 + RELAX_PER_TASK - 1) / RELAX_PER_TASK ))  # ceil(20 / relax_per_task)
JOBS_PER_PAIR=$((1 + DOCKING_ARRAYS + RELAX_TASKS_PER_PAIR))  # repack + docking tasks + relax tasks
MAX_CONCURRENT=$(( 999 / JOBS_PER_PAIR ))
if [[ $BATCH_SIZE -gt $MAX_CONCURRENT ]]; then
    echo ""
    echo "  ⚠ WARNING: batch_size=$BATCH_SIZE may exceed 999 job limit"
    echo "    Each pair needs ~$JOBS_PER_PAIR SLURM tasks"
    echo "    Max safe batch size: $MAX_CONCURRENT"
    echo "    Consider: --batch-size $MAX_CONCURRENT"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Split into batches (or use as-is) ──────────────────────────────────────
if [[ "$NO_SPLIT" == "true" ]]; then
    BATCH_FILES=("$TIER_CSV")
else
    echo "Splitting into batches of $BATCH_SIZE..."
    BATCH_OUTPUT=$(python "$BUILD_PAIRS" --split-batches "$TIER_CSV" --batch-size "$BATCH_SIZE" 2>&1)
    echo "$BATCH_OUTPUT"
    echo ""

    # Collect batch files
    BATCH_DIR=$(dirname "$TIER_CSV")
    BATCH_FILES=()
    for f in "${BATCH_DIR}"/${TIER_NAME}_batch*.csv; do
        if [[ -f "$f" ]]; then
            BATCH_FILES+=("$f")
        fi
    done
fi

echo "Running ${#BATCH_FILES[@]} batch(es)..."
echo ""

# ── Run orchestrator on each batch ──────────────────────────────────────────
for batch_csv in "${BATCH_FILES[@]}"; do
    batch_name=$(basename "$batch_csv" .csv)
    batch_cache="${CACHE_DIR}/${batch_name}"
    n_pairs=$(tail -n +2 "$batch_csv" | wc -l)

    echo "── Batch: $batch_name ($n_pairs pairs) ──"

    # Build orchestrator command
    CMD=(
        python "$ORCHESTRATOR"
        --pairs-csv "$batch_csv"
        --cache-dir "$batch_cache"
        --template-pdb "$TEMPLATE_PDB"
        --reference-pdb "$REFERENCE_PDB"
        --use-slurm
        --docking-repeats "$DOCKING_REPEATS"
        --docking-arrays "$DOCKING_ARRAYS"
        --workers "$WORKERS"
        --relax-per-task "$RELAX_PER_TASK"
    )

    if [[ "$SKIP_AF3" == "true" ]]; then
        CMD+=(--skip-af3)
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute:"
        echo "    ${CMD[*]}"
        echo ""
    else
        echo "  Running: ${CMD[*]}"
        "${CMD[@]}" 2>&1 | while IFS= read -r line; do echo "    $line"; done
        echo ""
    fi
done

echo "════════════════════════════════════════════════════════════════"
echo "  Done. Re-run this script after SLURM jobs complete to"
echo "  advance to the next stage."
echo ""
echo "  Check SLURM queue:  squeue -u \$USER"
echo "  Check job status:   sacct -j <jobid> --format=JobID,State,Elapsed"
echo "════════════════════════════════════════════════════════════════"
