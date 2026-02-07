#!/bin/bash
#SBATCH --job-name=ligand_search
#SBATCH --output=/projects/ryde3462/software/LigandMPNN/ligand_alignment_mpnn/logs/output_%A_%a.out
#SBATCH --error=/projects/ryde3462/software/LigandMPNN/ligand_alignment_mpnn/logs/error_%A_%a.err
#SBATCH --time=1:50:00
#SBATCH --array=0-999
#SBATCH --partition=amilan

HOME_DIR="/projects/ryde3462/bile_acids/CA"
mkdir -p ${HOME_DIR}/logs
cd ${HOME_DIR}

# Run the Python script, passing the config file and the SLURM array index
python /projects/ryde3462/software/ligand_alignment/grade_conformers_glycine_shaved_docking_multiple_slurm.py ${HOME_DIR}/config_multiple.txt ${SLURM_ARRAY_TASK_ID}

