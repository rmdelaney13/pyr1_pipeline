#!/bin/bash
#SBATCH --job-name=LigandMPNN_CDCA_boltz      # Job name
#SBATCH --output=LigandMPNN_CDCA_boltz_%A_%a.out
#SBATCH --error=LigandMPNN_CDCA_boltz_%A_%a.err
#SBATCH --partition=amilan
#SBATCH --account=ucb472_asc2
#SBATCH --ntasks=1                             # Run a single task
#SBATCH --qos=normal
#SBATCH --mem=8G                              # Total memory per task
#SBATCH --time=00:20:00                        # Time limit hrs:min:sec
#SBATCH --array=1-__ARRAY_COUNT__

# ------------------------------
# Load Required Modules
# ------------------------------
module load anaconda
conda activate ligandmpnn_env

# ------------------------------
# Define Variables
# ------------------------------
PDB_DIR="__PDB_DIR__"
OUTPUT_BASE="__OUTPUT_BASE__"
MODEL_SCRIPT="/projects/ryde3462/software/LigandMPNN/run.py"

# CDCA-specific configs (Boltz full-sequence numbering, +2 offset for positions >=72)
OMIT_JSON="/projects/ryde3462/pyr1_pipeline/design/mpnn/CDCA_omit_boltz.json"
BIAS_JSON="/projects/ryde3462/pyr1_pipeline/design/mpnn/CDCA_bias_boltz.json"

# ------------------------------
# Generate List of PDB Files
# ------------------------------
mapfile -t pdb_files < <(ls "${PDB_DIR}"/*.pdb | sort)
TOTAL_FILES=${#pdb_files[@]}
echo "Total number of PDB files: ${TOTAL_FILES}"

# ------------------------------
# Determine Batch Size per Job
# ------------------------------
GROUP_SIZE=1

START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * GROUP_SIZE ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * GROUP_SIZE - 1 ))
if [ $END_INDEX -ge $TOTAL_FILES ]; then
    END_INDEX=$(( TOTAL_FILES - 1 ))
fi

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing files from index ${START_INDEX} to ${END_INDEX}"

# ------------------------------
# Loop over the Selected Files
# ------------------------------
for (( i=START_INDEX; i<=END_INDEX; i++ )); do
    PDB_FILE="${pdb_files[$i]}"
    PDB_BASENAME=$(basename "${PDB_FILE}" .pdb)
    OUT_FOLDER="${OUTPUT_BASE}/${PDB_BASENAME}_mpnn"

    echo "Processing file: ${PDB_FILE}"
    mkdir -p "${OUT_FOLDER}"

    # ------------------------------
    # Run LigandMPNN for this file
    # ------------------------------
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

    echo "LigandMPNN processing completed for ${PDB_FILE}. Output saved to ${OUT_FOLDER}."
done

echo "All files in batch (job ${SLURM_ARRAY_TASK_ID}) have been processed."
