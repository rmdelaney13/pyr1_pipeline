#!/bin/bash
#SBATCH --job-name=boltz_test10
#SBATCH --partition=atesting_a100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=00:30:00
#SBATCH --output=boltz_test10_%j.out
#SBATCH --error=boltz_test10_%j.err
# Usage:
#   sbatch slurm/test_boltz_batch.sh <manifest_file> <out_dir>
#
# Runs first 10 YAMLs from manifest on the testing partition.

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

MANIFEST="$1"
OUT_DIR="${2:-boltz_test_output}"

if [ -z "$MANIFEST" ]; then
    echo "ERROR: Must provide manifest file as first argument"
    exit 1
fi

echo "Running first 10 predictions from $MANIFEST"

FAILED=0
DONE=0

for LINE_NUM in $(seq 1 10); do
    YAML_PATH=$(sed -n "${LINE_NUM}p" "$MANIFEST")
    if [ -z "$YAML_PATH" ]; then
        break
    fi

    YAML_NAME=$(basename "$YAML_PATH" .yaml)
    echo "--- [$DONE] Predicting: $YAML_NAME ---"

    boltz predict "$YAML_PATH" \
        --out_dir "$OUT_DIR" \
        --cache /projects/ryde3462/software/boltz_cache \
        --recycling_steps 3 \
        --diffusion_samples 5 \
        --output_format pdb

    if [ $? -ne 0 ]; then
        echo "FAILED: $YAML_NAME"
        FAILED=$((FAILED + 1))
    fi
    DONE=$((DONE + 1))
done

echo "Completed: $DONE predictions, $FAILED failures"
