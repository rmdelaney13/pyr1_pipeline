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
# The output CIF/PDB can then be used as template for all mutant predictions.
# Also saves the MSA .a3m for reuse.

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

PIPE_ROOT="/projects/ryde3462/software/pyr1_pipeline"
CACHE="/projects/ryde3462/software/boltz_cache"
OUT_DIR="/scratch/alpine/ryde3462/boltz_lca/wt_prediction"

mkdir -p "$OUT_DIR"

# Generate WT YAML using Python (guarantees clean ASCII sequence)
python -c "
from scripts.prepare_boltz_yamls import WT_PYR1_SEQUENCE, POCKET_RESIDUES, generate_yaml

yaml = generate_yaml(
    name='pyr1_wt_lca',
    sequence=WT_PYR1_SEQUENCE,
    ligand_smiles='CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C',
    mode='binary',
    msa_path=None,  # no msa: empty — let Boltz generate MSA
    template_path=None,
    pocket_constraint=True,
    affinity=True,
)
# Remove 'msa: empty' line so Boltz generates MSA
yaml = yaml.replace('      msa: empty\n', '')
with open('$OUT_DIR/pyr1_wt_lca.yaml', 'w') as f:
    f.write(yaml)
print('Generated WT YAML')
print(f'Sequence length: {len(WT_PYR1_SEQUENCE)}')
"

echo "YAML content:"
cat "$OUT_DIR/pyr1_wt_lca.yaml"
echo ""

echo "===== Predicting WT PYR1 + LCA (MSA from ColabFold) ====="
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
echo "Output structure (use as template for mutants):"
find "$OUT_DIR" -name "*.pdb" -path "*/predictions/*" 2>/dev/null
echo ""
echo "MSA (reuse for all mutant predictions):"
find "$OUT_DIR" -name "*.a3m" 2>/dev/null
echo ""
echo "Confidence JSON:"
find "$OUT_DIR" -name "confidence*" 2>/dev/null
