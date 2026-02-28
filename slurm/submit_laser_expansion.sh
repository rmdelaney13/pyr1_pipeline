#!/bin/bash
#SBATCH --job-name=laser_exp
#SBATCH --partition=aa100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=38400M
#SBATCH --time=01:00:00
#SBATCH --output=laser_exp_%A_%a.out
#SBATCH --error=laser_exp_%A_%a.err
# Usage:
#   sbatch --array=0-N submit_laser_expansion.sh <manifest> <output_dir> [batch_size] [designs_per_input]
#
# Each array task processes <batch_size> PDBs from the manifest, split across 2 GPUs.
# LASErMPNN run_batch_inference.py accepts a .txt file listing PDB paths.

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate /projects/ryde3462/software/envs/lasermpnn

LASER_ROOT="/projects/ryde3462/software/LASErMPNN"
export PYTHONPATH="${LASER_ROOT%/*}:${PYTHONPATH}"

MANIFEST="$1"
OUTPUT_DIR="${2:-laser_output}"
BATCH_SIZE="${3:-50}"
DESIGNS_PER_INPUT="${4:-3}"

if [ -z "$MANIFEST" ]; then
    echo "ERROR: Must provide manifest file as first argument"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$MANIFEST")
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1 ))
END=$(( START + BATCH_SIZE - 1 ))
if [ "$END" -gt "$TOTAL_LINES" ]; then END="$TOTAL_LINES"; fi
if [ "$START" -gt "$TOTAL_LINES" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no work (start $START > total $TOTAL_LINES)"
    exit 0
fi

# Create batch list for this task
BATCH_LIST=$(mktemp /tmp/laser_batch_XXXX.txt)
sed -n "${START},${END}p" "$MANIFEST" > "$BATCH_LIST"
N_PDBS=$(wc -l < "$BATCH_LIST")

echo "Task $SLURM_ARRAY_TASK_ID: processing PDBs $START-$END of $TOTAL_LINES ($N_PDBS PDBs)"

# Split batch across 2 GPUs
MIDPOINT=$(( (N_PDBS + 1) / 2 ))
BATCH_GPU0=$(mktemp /tmp/laser_gpu0_XXXX.txt)
BATCH_GPU1=$(mktemp /tmp/laser_gpu1_XXXX.txt)
head -n "$MIDPOINT" "$BATCH_LIST" > "$BATCH_GPU0"
tail -n +"$((MIDPOINT + 1))" "$BATCH_LIST" > "$BATCH_GPU1"

N_GPU0=$(wc -l < "$BATCH_GPU0")
N_GPU1=$(wc -l < "$BATCH_GPU1")
echo "  GPU 0: $N_GPU0 PDBs, GPU 1: $N_GPU1 PDBs"

LASER_CMD="python ${LASER_ROOT}/run_batch_inference.py"
LASER_ARGS="$DESIGNS_PER_INPUT --fix_beta --use_water --sequence_temp 0.3 --first_shell_sequence_temp 0.5 -c --ala_budget 4 --gly_budget 0 --output_fasta --model_weights_path ${LASER_ROOT}/model_weights/laser_weights_0p1A_nothing_heldout.pt"

# Run both GPUs in parallel
$LASER_CMD "$BATCH_GPU0" "$OUTPUT_DIR" $LASER_ARGS --device cuda:0 &
PID0=$!

if [ "$N_GPU1" -gt 0 ]; then
    $LASER_CMD "$BATCH_GPU1" "$OUTPUT_DIR" $LASER_ARGS --device cuda:1 &
    PID1=$!
    wait $PID0 $PID1
else
    wait $PID0
fi

rm -f "$BATCH_LIST" "$BATCH_GPU0" "$BATCH_GPU1"
echo "Done: $N_PDBS PDBs processed across 2 GPUs"
