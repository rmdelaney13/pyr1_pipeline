#!/bin/bash
#SBATCH --job-name=boltz_test
#SBATCH --partition=atesting_a100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=00:30:00
#SBATCH --output=boltz_test_%j.out
#SBATCH --error=boltz_test_%j.err

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

# Quick GPU check
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Create test input
cat > /tmp/test_boltz_${SLURM_JOB_ID}.yaml << 'EOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: "MQIFVKTLTGKTITL"
      msa: empty
EOF

# Run prediction
boltz predict /tmp/test_boltz_${SLURM_JOB_ID}.yaml \
    --out_dir /tmp/boltz_test_${SLURM_JOB_ID} \
    --cache /projects/ryde3462/software/boltz_cache \
    --override

echo "Exit code: $?"
echo "Done."
