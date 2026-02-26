#!/bin/bash
#SBATCH --job-name=pyr1_msa
#SBATCH --partition=atesting_a100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=00:30:00
#SBATCH --output=pyr1_msa_%j.out
#SBATCH --error=pyr1_msa_%j.err
# Usage:
#   sbatch slurm/generate_pyr1_msa.sh
#
# Runs a single WT PYR1 + LCA prediction WITH MSA generation (no msa: empty).
# After completion, grab the .a3m from the output for reuse.
# The MSA will be in: pyr1_msa_output/boltz_results_*/msa/A.a3m

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

OUT_DIR="/scratch/alpine/ryde3462/boltz_lca/pyr1_msa_output"
YAML_PATH="/scratch/alpine/ryde3462/boltz_lca/pyr1_wt_msa_gen.yaml"

# Write a WT PYR1 YAML without msa: empty so Boltz generates the MSA
cat > "$YAML_PATH" << 'YAMLEOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSVЕQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRIWTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
  - ligand:
      id: B
      smiles: "CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C"
YAMLEOF

echo "Generating WT PYR1 MSA via Boltz data pipeline..."
echo "YAML: $YAML_PATH"
echo "Output: $OUT_DIR"

boltz predict "$YAML_PATH" \
    --out_dir "$OUT_DIR" \
    --cache /projects/ryde3462/software/boltz_cache \
    --recycling_steps 3 \
    --diffusion_samples 1 \
    --output_format pdb \
    --use_msa_server

echo "Done. Look for the .a3m file in:"
echo "  $OUT_DIR/boltz_results_*/msa/"
echo ""
echo "Then use it for all predictions with:"
echo "  --msa /path/to/A.a3m"
