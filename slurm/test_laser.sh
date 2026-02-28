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
conda activate /projects/ryde3462/software/envs/lasermpnn

LASER_ROOT="/projects/ryde3462/software/LASErMPNN"
export PYTHONPATH="${LASER_ROOT%/*}:${PYTHONPATH}"

echo "=== Environment ==="
which python
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Running LASErMPNN on test PDB ==="
python "${LASER_ROOT}/run_inference.py" \
    /scratch/alpine/ryde3462/laser_test/prepped/pair_0771_model_0.pdb \
    -o /scratch/alpine/ryde3462/laser_test/laser_out.pdb \
    --fix_beta \
    --device cuda:0

echo ""
echo "=== Output ==="
ls -la /scratch/alpine/ryde3462/laser_test/laser_out*
echo ""
echo "Done"
