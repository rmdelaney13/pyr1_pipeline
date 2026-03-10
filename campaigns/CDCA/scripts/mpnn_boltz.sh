#!/bin/bash
#SBATCH --job-name=LigandMPNN_CDCA_boltz
#SBATCH --output=LigandMPNN_CDCA_boltz_%A_%a.out
#SBATCH --error=LigandMPNN_CDCA_boltz_%A_%a.err
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=1-__ARRAY_COUNT__

module load anaconda
conda activate ligandmpnn_env

PDB_DIR="__PDB_DIR__"
OUTPUT_BASE="__OUTPUT_BASE__"
MODEL_SCRIPT="/projects/ryde3462/software/LigandMPNN/run.py"

OMIT_JSON="/projects/ryde3462/software/pyr1_pipeline/campaigns/CDCA/mpnn/omit_boltz.json"
BIAS_JSON="/projects/ryde3462/software/pyr1_pipeline/campaigns/CDCA/mpnn/bias_boltz.json"

mapfile -t pdb_files < <(ls "${PDB_DIR}"/*.pdb | sort)
TOTAL_FILES=${#pdb_files[@]}
echo "Total number of PDB files: ${TOTAL_FILES}"

GROUP_SIZE=1
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * GROUP_SIZE ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * GROUP_SIZE - 1 ))
if [ $END_INDEX -ge $TOTAL_FILES ]; then
    END_INDEX=$(( TOTAL_FILES - 1 ))
fi

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing files from index ${START_INDEX} to ${END_INDEX}"

for (( i=START_INDEX; i<=END_INDEX; i++ )); do
    PDB_FILE="${pdb_files[$i]}"
    PDB_BASENAME=$(basename "${PDB_FILE}" .pdb)
    OUT_FOLDER="${OUTPUT_BASE}/${PDB_BASENAME}_mpnn"

    echo "Processing file: ${PDB_FILE}"
    mkdir -p "${OUT_FOLDER}"

    cd /projects/ryde3462/software/LigandMPNN
    python "${MODEL_SCRIPT}" \
        --seed 111 \
        --model_type "ligand_mpnn" \
        --pdb_path "${PDB_FILE}" \
        --redesigned_residues "A59 A81 A83 A92 A94 A108 A110 A117 A120 A122 A141 A159 A160 A163 A164 A167" \
        --out_folder "${OUT_FOLDER}" \
        --number_of_batches 1 \
        --batch_size 20 \
        --temperature 0.3 \
        --omit_AA_per_residue "${OMIT_JSON}" \
        --bias_AA_per_residue "${BIAS_JSON}" \
        --pack_side_chains 0 \
        --checkpoint_ligand_mpnn "/projects/ryde3462/software/LigandMPNN/model_params/ligandmpnn_v_32_020_25.pt" \
        --pack_with_ligand_context 1

    echo "LigandMPNN completed for ${PDB_FILE}. Output: ${OUT_FOLDER}"
done

echo "All files in batch (job ${SLURM_ARRAY_TASK_ID}) have been processed."
