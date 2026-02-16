#!/bin/bash
#
# Test script for running 3 example pairs through the ML docking pipeline
# Examples: Win wt (abscisic acid), PYR1nitav2 (nitazene), lca seq16 (Lithocholic Acid)
#
# Usage (on cluster):
#   bash ml_modelling/scripts/test_three_pairs.sh

set -e  # Exit on error

echo "=========================================="
echo "ML Pipeline Test: 3 Example Pairs"
echo "=========================================="
echo ""

# ===== CONFIGURATION =====
# Adjust these paths for your cluster environment
PROJECT_ROOT="/path/to/pyr1_pipeline"  # UPDATE THIS!
CACHE_DIR="${SCRATCH}/ml_dataset_test"  # Or your preferred scratch location
TEMPLATE_PDB="${PROJECT_ROOT}/templates/PYR1_WT_clean.pdb"  # UPDATE THIS!

# Input CSV with 3 test pairs
TEST_CSV="${PROJECT_ROOT}/ml_modelling/data/test_three_pairs.csv"

# Check if test CSV exists
if [ ! -f "$TEST_CSV" ]; then
    echo "ERROR: Test CSV not found: $TEST_CSV"
    echo "Make sure you've created the test_three_pairs.csv file!"
    exit 1
fi

# Check if template PDB exists
if [ ! -f "$TEMPLATE_PDB" ]; then
    echo "ERROR: Template PDB not found: $TEMPLATE_PDB"
    echo "Please update TEMPLATE_PDB path in this script!"
    exit 1
fi

echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Test CSV: $TEST_CSV"
echo "  Template PDB: $TEMPLATE_PDB"
echo "  Cache dir: $CACHE_DIR"
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"

# ===== RUN PIPELINE =====
echo "Starting pipeline orchestration..."
echo ""

# Option 1: Run locally (for quick testing, no SLURM)
# This will run conformer generation and docking sequentially
python "${PROJECT_ROOT}/ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py" \
    --pairs-csv "$TEST_CSV" \
    --cache-dir "$CACHE_DIR" \
    --template-pdb "$TEMPLATE_PDB" \
    --docking-repeats 50 \
    --max-pairs 3

# Option 2: Submit to SLURM (uncomment below, comment above)
# python "${PROJECT_ROOT}/ml_modelling/scripts/orchestrate_ml_dataset_pipeline.py" \
#     --pairs-csv "$TEST_CSV" \
#     --cache-dir "$CACHE_DIR" \
#     --template-pdb "$TEMPLATE_PDB" \
#     --docking-repeats 50 \
#     --max-pairs 3 \
#     --use-slurm

echo ""
echo "=========================================="
echo "Pipeline orchestration complete!"
echo "=========================================="
echo ""
echo "Check results in: $CACHE_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify conformer generation: ls $CACHE_DIR/*/conformers/"
echo "  2. Check docking outputs: ls $CACHE_DIR/*/docking/"
echo "  3. View processing summary: cat $CACHE_DIR/processing_summary.csv"
echo ""
