#!/bin/bash
#SBATCH --job-name=boltz_pred
#SBATCH --partition=aa100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=04:00:00
#SBATCH --output=boltz_%A_%a.out
#SBATCH --error=boltz_%A_%a.err
# Usage:
#   sbatch --array=0-N submit_boltz.sh <manifest_file> <out_dir> <batch_size> <diffusion_samples> [extra_boltz_flags]
#
# Each array task processes <batch_size> YAMLs from the manifest.
# Array index range = 0 to ceil(total_yamls / batch_size) - 1
# diffusion_samples: default 5. For ternary without affinity, 5 is fine.
# If affinity is in the YAML and causing crashes, fall back to 1.
#
# Example (552 YAMLs, 20 per job = 28 array tasks):
#   sbatch --array=0-27 slurm/submit_boltz.sh manifest.txt boltz_output 20 5
#   sbatch --array=0-27 slurm/submit_boltz.sh manifest.txt boltz_output 20 1  # ternary

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

MANIFEST="$1"
OUT_DIR="${2:-boltz_output}"
BATCH_SIZE="${3:-20}"
DIFF_SAMPLES="${4:-5}"
EXTRA_FLAGS="${@:5}"

if [ -z "$MANIFEST" ]; then
    echo "ERROR: Must provide manifest file as first argument"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$MANIFEST")
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1 ))
END=$(( START + BATCH_SIZE - 1 ))

# Clamp END to total lines
if [ "$END" -gt "$TOTAL_LINES" ]; then
    END=$TOTAL_LINES
fi

if [ "$START" -gt "$TOTAL_LINES" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no work (start=$START > total=$TOTAL_LINES)"
    exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID: processing lines $START-$END of $TOTAL_LINES (batch_size=$BATCH_SIZE)"

FAILED=0
DONE=0

for LINE_NUM in $(seq $START $END); do
    YAML_PATH=$(sed -n "${LINE_NUM}p" "$MANIFEST")
    if [ -z "$YAML_PATH" ]; then
        continue
    fi

    YAML_NAME=$(basename "$YAML_PATH" .yaml)
    echo "--- [$DONE] Predicting: $YAML_NAME ---"

    boltz predict "$YAML_PATH" \
        --out_dir "$OUT_DIR" \
        --cache /projects/ryde3462/software/boltz_cache \
        --recycling_steps 3 \
        --diffusion_samples $DIFF_SAMPLES \
        --max_msa_seqs 32 \
        --output_format pdb \
        $EXTRA_FLAGS

    if [ $? -ne 0 ]; then
        echo "FAILED: $YAML_NAME"
        FAILED=$((FAILED + 1))
    fi
    DONE=$((DONE + 1))
done

echo "Completed: $DONE predictions, $FAILED failures"
