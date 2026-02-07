#!/bin/bash
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH --partition=amilan
#SBATCH --job-name=af3_msa_array
#SBATCH --output=af3_msa_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-86%20            # replace <N> with the count of JSONs

module purge
module load alphafold/3.0.0

# Directory containing all your mutant JSON inputs
INPUT_DIR=/scratch/alpine/ryde3462/cholic_acid_design/af3_inputs/
# Base output directory for the _data.json files
OUTPUT_ROOT=/scratch/alpine/ryde3462/cholic_acid_design/af3_inputs/msas
# Model parameters
AF3_MODEL_PARAMETERS_DIR=/projects/ryde3462/software/af3

cd $INPUT_DIR

# Pick the Nth JSON in lex order
JSON=$(ls *.json | sed -n ${SLURM_ARRAY_TASK_ID}p)
JOBNAME=${JSON%.json}

echo "MSA for $JSON → $OUTPUT_ROOT/$JOBNAME"
run_alphafold \
  --json_path=${INPUT_DIR}/${JSON} \
  --output_dir=${OUTPUT_ROOT}/${JOBNAME} \
  --model_dir=${AF3_MODEL_PARAMETERS_DIR} \
  --norun_inference

