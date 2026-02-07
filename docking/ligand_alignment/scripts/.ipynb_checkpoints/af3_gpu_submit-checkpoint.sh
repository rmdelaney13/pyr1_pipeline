#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --partition=atesting_a100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --job-name=af3_input_dir
#SBATCH --output=af3_input_dir_%j.out
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --account=ucb472_asc2

module purge
module load alphafold/3.0.0

# Parent folder containing one sub-folder per target, each with its feature files:
export INPUT_DIR=/scratch/alpine/ryde3462/cholic_acid_design/af3_inputs/msas/batch_001

# Where to dump all of the per-target inference outputs:
export OUTPUT_DIR=/scratch/alpine/ryde3462/cholic_acid_design/af3_output

# Where your AF3 model params live:
export AF3_MODEL_PARAMETERS_DIR=/projects/ryde3462/software/af3

mkdir -p $OUTPUT_DIR

run_alphafold \
  --input_dir=$INPUT_DIR \
  --output_dir=$OUTPUT_DIR \
  --model_dir=$AF3_MODEL_PARAMETERS_DIR \
  --norun_data_pipeline

