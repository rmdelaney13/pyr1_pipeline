#!/bin/bash
#SBATCH --job-name=ml_relax
#SBATCH --output=ml_relax_%A_%a.out
#SBATCH --error=ml_relax_%A_%a.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=ucb472_asc2
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# SLURM array wrapper for ML pipeline relax stage.
# Each array task processes BATCH_SIZE rows from the manifest file,
# running relax_general_universal.py sequentially for each row.
#
# Usage: sbatch --array=1-N submit_relax_ml.sh /path/to/relax_manifest.tsv [BATCH_SIZE]
#
# Manifest format (tab-separated, one row per structure):
#   input_pdb  output_pdb  ligand_params  xml_path  ligand_chain  water_chain
#
# BATCH_SIZE (default: 1) controls how many manifest rows each array task processes.
# Array task 1 processes rows 1..BATCH_SIZE, task 2 processes rows BATCH_SIZE+1..2*BATCH_SIZE, etc.

# Change to the directory where sbatch was called (project root)
cd "$SLURM_SUBMIT_DIR" || exit 1

MANIFEST="$1"
BATCH_SIZE="${2:-1}"

if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest file not found: $MANIFEST"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$MANIFEST")

# Calculate row range for this array task (1-indexed)
START_ROW=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + 1 ))
END_ROW=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
if [ "$END_ROW" -gt "$TOTAL_LINES" ]; then
    END_ROW="$TOTAL_LINES"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: Processing manifest rows ${START_ROW}-${END_ROW} (batch_size=${BATCH_SIZE})"

FAILURES=0

for ROW_NUM in $(seq "$START_ROW" "$END_ROW"); do
    LINE=$(sed -n "${ROW_NUM}p" "$MANIFEST")

    if [ -z "$LINE" ]; then
        echo "  Row ${ROW_NUM}: empty, skipping"
        continue
    fi

    IFS=$'\t' read -r INPUT_PDB OUTPUT_PDB LIGAND_PARAMS XML_PATH LIG_CHAIN WAT_CHAIN <<< "$LINE"

    echo "  Row ${ROW_NUM}: Relaxing ${INPUT_PDB}"
    echo "    Output: ${OUTPUT_PDB}"

    python design/rosetta/relax_general_universal.py \
        "$INPUT_PDB" "$OUTPUT_PDB" "$LIGAND_PARAMS" \
        --xml_path "$XML_PATH" \
        --ligand_chain "$LIG_CHAIN" \
        --water_chain "$WAT_CHAIN"

    RC=$?
    echo "    Row ${ROW_NUM}: exit code ${RC}"
    if [ "$RC" -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
done

echo "Task ${SLURM_ARRAY_TASK_ID}: Done. Processed rows ${START_ROW}-${END_ROW}, failures=${FAILURES}"
exit $FAILURES
