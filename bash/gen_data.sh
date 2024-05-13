#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=jazzy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/fuels/%x-%j.out
#SBATCH --error=slurm_run/fuels/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

python3 generate_data.py 