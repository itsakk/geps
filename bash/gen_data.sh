#!/bin/bash
#SBATCH --job-name=xxx
#SBATCH --partition=xxx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/xxx/%x-%j.out
#SBATCH --error=slurm_run/xxx/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate geps

python3 geps/datasets/combined.py 
