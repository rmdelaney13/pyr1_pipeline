#!/bin/bash
#
# SLURM array job wrapper for mutant docking
#
# This script is called by orchestrate_ml_dataset_pipeline.py for parallel docking
# using SLURM array jobs.
#
# Usage:
#   sbatch --array=0-9 submit_docking_mutant.sh <config_file>
#
# Arguments:
#   $1 - Path to docking config file
#
# Environment variables:
#   SLURM_ARRAY_TASK_ID - Array task index (0-based)

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "ERROR: Config file path required"
    echo "Usage: submit_docking_mutant.sh <config_file>"
    exit 1
fi

CONFIG_FILE="$1"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Mutant Docking - Array Task $ARRAY_INDEX"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Array index: $ARRAY_INDEX"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=========================================="
echo ""

# Run docking script with array index
python docking/scripts/grade_conformers_mutant_docking.py "$CONFIG_FILE" "$ARRAY_INDEX"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Array task $ARRAY_INDEX completed successfully"
else
    echo ""
    echo "✗ Array task $ARRAY_INDEX failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
