#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/foca/%x-%j.out
#SBATCH --error=slurm_run/foca/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='pendulum'
batch_size_train=1
batch_size_val=16
epochs=10
inner_lr=1.0
outer_lr=0.001
inner_steps=100
seed=42
hidden_c=64
state_c=2
ctx_dim=2
tau=0.1
run_name='astral-lake-2268'

# burgers morning-bird-1374
# gs warm-gorge-1373
# kolmo azure-star-1387
# pendulum sunny-yogurt-1947

python3 baselines/FOCA/foca_adapt.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.inner_lr=$inner_lr" "optim.outer_lr=$outer_lr" "model.hidden_c=$hidden_c" "model.state_c=$state_c" "optim.init_type=$init_type" "optim.inner_steps=$inner_steps" "optim.test_inner_steps=$test_inner_steps" "model.ctx_dim=$ctx_dim" "optim.tau=$tau" "pretrain.run_name=$run_name"