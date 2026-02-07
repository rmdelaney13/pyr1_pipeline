#!/bin/bash
#SBATCH --job-name=ligand_docking
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=1-900
#SBATCH --qos=normal
#SBATCH --account=ucb-general
#SBATCH --partition=amilan

HOME_DIR="/projects/ryde3462/bile_acids/CA/"
mkdir -p ${HOME_DIR}/logs
cd ${HOME_DIR}

# Run the Python script, passing the config file and the SLURM array index
python /projects/ryde3462/software/ligand_alignment/grade_conformers_docked_to_sequence_multiple_slurm1.py ${HOME_DIR}/config_multiple_CA.txt ${SLURM_ARRAY_TASK_ID}


