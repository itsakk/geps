#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/new_ver/%x-%j.out
#SBATCH --error=slurm_run/new_ver/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='gs' # pendulum, burgers, gs, lv
batch_size_train=1 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=32
epochs=20000
lr=0.01
seed=123
hidden_c=64
state_c=2 # 2 others, 1 if burgers
code_c=2
is_complete=complete
type_augment=serie
factor=1 # 1 if lv or pendulum, 0.0005 if gs
run_name='summer-morning-1754'

python3 adapt.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "model.code_c=$code_c" "pretrain.run_name=$run_name" "model.factor=$factor" "model.is_complete=$is_complete" "model.type_augment=$type_augment"