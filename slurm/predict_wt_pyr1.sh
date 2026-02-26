#!/bin/bash
#SBATCH --job-name=boltz_wt
#SBATCH --partition=atesting_a100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=01:00:00
#SBATCH --output=boltz_wt_%j.out
#SBATCH --error=boltz_wt_%j.err
# Usage:
#   sbatch slurm/predict_wt_pyr1.sh
#
# Predicts WT PYR1 + LCA from scratch (with MSA, no template).
# The output CIF can then be used as template for all mutant predictions.
# Also saves the MSA .a3m for reuse.

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

CACHE="/projects/ryde3462/software/boltz_cache"
OUT_DIR="/scratch/alpine/ryde3462/boltz_lca/wt_prediction"

mkdir -p "$OUT_DIR"

# WT PYR1 sequence (3QN1 stabilized construct, 181 aa)
cat > "$OUT_DIR/pyr1_wt_lca.yaml" << 'YAMLEOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: "MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSVЕQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRIWTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
  - ligand:
      id: B
      smiles: "CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C"
constraints:
  - pocket:
      binder: B
      contacts:
        - [A, 59]
        - [A, 62]
        - [A, 79]
        - [A, 81]
        - [A, 83]
        - [A, 88]
        - [A, 90]
        - [A, 92]
        - [A, 106]
        - [A, 108]
        - [A, 110]
        - [A, 115]
        - [A, 116]
        - [A, 118]
        - [A, 120]
        - [A, 139]
        - [A, 157]
        - [A, 158]
        - [A, 159]
        - [A, 161]
        - [A, 162]
        - [A, 165]
      max_distance: 6.0
properties:
  - affinity:
      binder: B
YAMLEOF

echo "Predicting WT PYR1 + LCA (with MSA from ColabFold server)..."
boltz predict "$OUT_DIR/pyr1_wt_lca.yaml" \
    --out_dir "$OUT_DIR" \
    --cache "$CACHE" \
    --recycling_steps 3 \
    --diffusion_samples 5 \
    --output_format pdb \
    --use_potentials \
    --use_msa_server

echo "Exit code: $?"
echo ""
echo "Output structure (use as template):"
ls "$OUT_DIR"/boltz_results_*/predictions/
echo ""
echo "MSA (reuse for mutant predictions):"
ls "$OUT_DIR"/boltz_results_*/msa/
