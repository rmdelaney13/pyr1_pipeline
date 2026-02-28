#!/bin/bash
#SBATCH --job-name=test_laser
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --account=ucb472_asc2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=test_laser_%j.out
#SBATCH --error=test_laser_%j.err

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate /projects/ryde3462/software/envs/lasermpnn

LASER_ROOT="/projects/ryde3462/software/LASErMPNN"
export PYTHONPATH="${LASER_ROOT%/*}:${PYTHONPATH}"

TEST_DIR="/scratch/alpine/ryde3462/laser_test"
TEST_PDB="${TEST_DIR}/prepped/pair_0771_model_0.pdb"

echo "=== Environment ==="
which python
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Create manifest for batch inference (single PDB)
MANIFEST="${TEST_DIR}/test_manifest.txt"
echo "$TEST_PDB" > "$MANIFEST"

echo ""
echo "=== Running LASErMPNN batch inference (with --use_water) ==="
python "${LASER_ROOT}/run_batch_inference.py" \
    "$MANIFEST" \
    "${TEST_DIR}/batch_output" \
    3 \
    --fix_beta \
    --use_water \
    --sequence_temp 0.3 \
    --first_shell_sequence_temp 0.5 \
    -c \
    --ala_budget 4 \
    --gly_budget 0 \
    --output_fasta \
    --device cuda:0 \
    --model_weights_path "${LASER_ROOT}/model_weights/laser_weights_0p1A_nothing_heldout.pt"

echo ""
echo "=== Output files ==="
find "${TEST_DIR}/batch_output" -type f | head -20

echo ""
echo "=== FASTA output ==="
cat "${TEST_DIR}/batch_output"/*/designs.fasta 2>/dev/null || echo "No FASTA found"

echo ""
echo "Done"
