#!/bin/bash
#SBATCH --job-name=boltz_tern
#SBATCH --partition=atesting_a100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=01:00:00
#SBATCH --output=boltz_tern_%j.out
#SBATCH --error=boltz_tern_%j.err
# Usage:
#   sbatch slurm/predict_wt_ternary.sh
#
# Predicts WT PYR1 + LCA + HAB1 ternary complex from scratch (with MSA).
# Output CIF can be used as template for ternary mutant predictions.

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

PIPE_ROOT="/projects/ryde3462/software/pyr1_pipeline"
CACHE="/projects/ryde3462/software/boltz_cache"
OUT_DIR="/scratch/alpine/ryde3462/boltz_lca/wt_ternary"

mkdir -p "$OUT_DIR"

# Full-length HAB1 sequence for ternary complex
HAB1="GAMGRSVYELDCIPLWGTVSIQGNRSEMEDAFAVSPHFLKLPIKMLMGDHEGMSPSLTHLTGHFFGVYDGHGGHKVADYCRDRLHFALAEEIERIKDELCKRNTGEGRQVQWDKVFTSCFLTVDGEIEGKIGRAVVGSSDKVLEAVASETVGSTAVVALVCSSHIVVSNCGDSRAVLFRGKEAMPLSVDHKPDREDEYARIENAGGKVIQWQGARVFGVLAMSRSIGDRYLKPYVIPEPEVTFMPRSREDECLILASDGLWDVMNNQEVCEIARRRILMWHKKNGAPPLAERGKGIDPACQAAADYLSMLALQKGSKDNISIIVIDLKAQRKFKTRT"

# Generate ternary YAML using Python (clean ASCII sequence)
python -c "
from scripts.prepare_boltz_yamls import WT_PYR1_SEQUENCE

hab1 = '${HAB1}'
lca = 'CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C'

lines = ['version: 1', 'sequences:']

# PYR1 (chain A)
lines.append('  - protein:')
lines.append('      id: A')
lines.append(f'      sequence: \"{WT_PYR1_SEQUENCE}\"')

# LCA (chain B)
lines.append('  - ligand:')
lines.append('      id: B')
lines.append(f'      smiles: \"{lca}\"')

# HAB1 (chain C)
lines.append('  - protein:')
lines.append('      id: C')
lines.append(f'      sequence: \"{hab1}\"')

yaml = '\n'.join(lines) + '\n'
with open('$OUT_DIR/pyr1_wt_lca_hab1.yaml', 'w') as f:
    f.write(yaml)
print('Generated ternary YAML')
print(f'PYR1: {len(WT_PYR1_SEQUENCE)} aa')
print(f'HAB1: {len(hab1)} aa')
"

echo "YAML content:"
cat "$OUT_DIR/pyr1_wt_lca_hab1.yaml"
echo ""

echo "===== Predicting WT PYR1 + LCA + HAB1 ternary (MSA from ColabFold) ====="
boltz predict "$OUT_DIR/pyr1_wt_lca_hab1.yaml" \
    --out_dir "$OUT_DIR" \
    --cache "$CACHE" \
    --recycling_steps 3 \
    --diffusion_samples 5 \
    --output_format pdb \
    --use_msa_server

echo "Exit code: $?"
echo ""
echo "Output structure:"
find "$OUT_DIR" -name "*.pdb" -path "*/predictions/*" 2>/dev/null
echo ""
echo "MSA files:"
find "$OUT_DIR" -name "*.a3m" 2>/dev/null
echo ""
echo "Confidence:"
find "$OUT_DIR" -name "confidence*" 2>/dev/null
