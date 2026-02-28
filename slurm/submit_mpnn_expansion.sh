#!/bin/bash
#SBATCH --job-name=mpnn_exp
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=mpnn_exp_%A_%a.out
#SBATCH --error=mpnn_exp_%A_%a.err
# Usage:
#   sbatch --array=1-N submit_mpnn_expansion.sh <manifest> <output_dir>
#
# Each array task processes one PDB from the manifest.
# Runs LigandMPNN with 3 new sequences per PDB at the 16 Boltz-numbered
# pocket positions, with light omit (A59, A159) and K bias at A141.

cd "$SLURM_SUBMIT_DIR"

module load anaconda
conda activate ligandmpnn_env

MANIFEST="$1"
OUTPUT_DIR="${2:-mpnn_output}"

if [ -z "$MANIFEST" ]; then
    echo "ERROR: Must provide manifest file as first argument"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$MANIFEST")
LINE_NUM=$SLURM_ARRAY_TASK_ID

if [ "$LINE_NUM" -gt "$TOTAL_LINES" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no work (line $LINE_NUM > total $TOTAL_LINES)"
    exit 0
fi

PDB_FILE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
if [ -z "$PDB_FILE" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: empty line $LINE_NUM"
    exit 0
fi

PDB_BASENAME=$(basename "$PDB_FILE" .pdb)
OUT_FOLDER="${OUTPUT_DIR}/${PDB_BASENAME}_mpnn"
mkdir -p "$OUT_FOLDER"

echo "Task $SLURM_ARRAY_TASK_ID: processing $PDB_BASENAME"

# Paths
MPNN_ROOT="/projects/ryde3462/software/LigandMPNN"
PIPE_ROOT="/projects/ryde3462/software/pyr1_pipeline"

cd "$MPNN_ROOT"
python "${MPNN_ROOT}/run.py" \
    --seed 111 \
    --model_type "ligand_mpnn" \
    --pdb_path "$PDB_FILE" \
    --redesigned_residues "A59 A81 A83 A92 A94 A108 A110 A117 A120 A122 A141 A159 A160 A163 A164 A167" \
    --out_folder "$OUT_FOLDER" \
    --number_of_batches 1 \
    --batch_size 3 \
    --temperature 0.3 \
    --omit_AA_per_residue "${PIPE_ROOT}/design/mpnn/expansion_omit.json" \
    --bias_AA_per_residue "${PIPE_ROOT}/design/mpnn/expansion_bias.json" \
    --pack_side_chains 0 \
    --checkpoint_ligand_mpnn "${MPNN_ROOT}/model_params/ligandmpnn_v_32_020_25.pt" \
    --pack_with_ligand_context 1

echo "Done: $PDB_BASENAME -> $OUT_FOLDER"
