#!/bin/bash
#SBATCH --job-name=boltz_ab
#SBATCH --partition=atesting_a100
#SBATCH --account=ucb472_asc2
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=38400M
#SBATCH --time=01:00:00
#SBATCH --output=boltz_ab_%j.out
#SBATCH --error=boltz_ab_%j.err
# Usage:
#   sbatch slurm/test_boltz_ab.sh
#
# A/B test: same WT PYR1 + LCA predicted two ways:
#   A) msa: empty + template (no evolutionary info)
#   B) --use_msa_server + --max_msa_seqs 32 + template (shallow MSA)
#
# First converts PDB template to Boltz-compatible CIF with proper metadata.

cd "$SLURM_SUBMIT_DIR"

module load anaconda cuda/12.1.1
source activate boltz_env

PIPE_ROOT="/projects/ryde3462/software/pyr1_pipeline"
PDB_TEMPLATE="${PIPE_ROOT}/structures/templates/Pyr1_LCA_mutant_template.pdb"
CACHE="/projects/ryde3462/software/boltz_cache"
OUT_BASE="/scratch/alpine/ryde3462/boltz_lca/ab_test_v2"
CIF_TEMPLATE="${OUT_BASE}/Pyr1_LCA_template_boltz.cif"

mkdir -p "$OUT_BASE"

# Step 0: Convert PDB to proper Boltz-compatible CIF
echo "===== Converting PDB to Boltz CIF ====="
python "${PIPE_ROOT}/scripts/pdb_to_boltz_cif.py" "$PDB_TEMPLATE" "$CIF_TEMPLATE"
if [ $? -ne 0 ]; then
    echo "FATAL: CIF conversion failed"
    exit 1
fi
echo ""

# WT PYR1 sequence (safe ASCII, no Unicode issues)
SEQUENCE="MASELTPEERSELKNSIAEFHTYQLDPGSCSSLHAQRIHAPPELVWSIVRRFDKPQTYKHFIKSCSVЕQNFEMRVGCTRDVIVISGLPANTSTERLDILDDERRVTGFSIIGGEHRLTNYKSVTTVHRFEKENRIWTVVLESYVVDMPEGNSEDDTRMFADTVVKLNLQKLATVAEAMARN"
LCA_SMILES="CC(CCC(=O)O)C1CCC2C1(CCC3C2CCC4C3(CCC(C4)O)C)C"

# ── Test A: no MSA (msa: empty) + CIF template ──
cat > "$OUT_BASE/test_noMSA.yaml" << YAMLEOF
version: 1
sequences:
  - protein:
      id: A
      sequence: "${SEQUENCE}"
      msa: empty
  - ligand:
      id: B
      smiles: "${LCA_SMILES}"
templates:
  - cif: ${CIF_TEMPLATE}
    chain_id: A
    force: true
    threshold: 2.0
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

# ── Test B: shallow MSA (auto-generated, capped at 32) + CIF template ──
cat > "$OUT_BASE/test_msa32.yaml" << YAMLEOF
version: 1
sequences:
  - protein:
      id: A
      sequence: "${SEQUENCE}"
  - ligand:
      id: B
      smiles: "${LCA_SMILES}"
templates:
  - cif: ${CIF_TEMPLATE}
    chain_id: A
    force: true
    threshold: 2.0
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

echo "===== TEST A: no MSA + CIF template ====="
boltz predict "$OUT_BASE/test_noMSA.yaml" \
    --out_dir "$OUT_BASE/out_noMSA" \
    --cache "$CACHE" \
    --recycling_steps 3 \
    --diffusion_samples 5 \
    --output_format pdb \
    --use_potentials
echo "Test A exit code: $?"

echo ""
echo "===== TEST B: MSA (max 32 seqs) + CIF template ====="
boltz predict "$OUT_BASE/test_msa32.yaml" \
    --out_dir "$OUT_BASE/out_msa32" \
    --cache "$CACHE" \
    --recycling_steps 3 \
    --diffusion_samples 5 \
    --max_msa_seqs 32 \
    --output_format pdb \
    --use_potentials \
    --use_msa_server
echo "Test B exit code: $?"

echo ""
echo "===== DONE ====="
echo "Compare structures:"
echo "  A (no MSA):    $OUT_BASE/out_noMSA/"
echo "  B (MSA 32):    $OUT_BASE/out_msa32/"
