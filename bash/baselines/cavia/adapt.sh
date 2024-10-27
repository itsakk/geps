#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/cavia/%x-%j.out
#SBATCH --error=slurm_run/cavia/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='pendulum'
batch_size_train=1
batch_size_val=32
epochs=10
inner_lr=0.1
outer_lr=0.001
inner_steps=100
seed=123
hidden_c=64
state_c=2
ctx_dim=2
run_name='deep-bush-2267'

python3 baselines/CAVIA/cavia_adapt.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.inner_lr=$inner_lr" "optim.outer_lr=$outer_lr" "model.hidden_c=$hidden_c" "model.state_c=$state_c" "optim.init_type=$init_type" "optim.inner_steps=$inner_steps" "optim.test_inner_steps=$test_inner_steps" "model.ctx_dim=$ctx_dim" "pretrain.run_name=$run_name"

# generous-jazz-1317 gs
# hopeful-totem-1348 burgers
# autumn-dust-1386 kolmo
# dauntless-sound-1943 pendulum