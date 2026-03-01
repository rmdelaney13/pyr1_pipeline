#!/bin/bash
#SBATCH --job-name=hab1_msa
#SBATCH --partition=aa100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=00:20:00
#SBATCH --output=hab1_msa_%j.out
#SBATCH --error=hab1_msa_%j.err
# Usage:
#   cd /projects/ryde3462/software/pyr1_pipeline
#   sbatch slurm/generate_hab1_msa.sh
#
# Generates a standalone HAB1 .a3m MSA by running a quick single-chain
# Boltz prediction with --use_msa_server. The resulting .a3m can then
# be used with prepare_boltz_yamls.py --hab1-msa for batch ternary runs.
#
# Output: /scratch/alpine/ryde3462/boltz_lca/hab1_msa/hab1.a3m

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

PIPE_ROOT="/projects/ryde3462/software/pyr1_pipeline"
OUT_DIR="/scratch/alpine/ryde3462/boltz_lca/hab1_msa"

python "${PIPE_ROOT}/scripts/extract_chain_msa.py" \
    --out-dir "${OUT_DIR}" \
    --name hab1 \
    --run-boltz

echo ""
echo "Exit code: $?"
echo ""
echo "HAB1 MSA output:"
ls -la "${OUT_DIR}/hab1.a3m" 2>/dev/null || echo "NOT FOUND"
echo ""
echo "Sequence count:"
grep -c "^>" "${OUT_DIR}/hab1.a3m" 2>/dev/null || echo "N/A"
