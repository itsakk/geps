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

dataset_name='gs' # pendulum, burgers, gs, lv
batch_size_train=8 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=16
epochs=20000
lr=0.01
seed=123
hidden_c=64
state_c=2 # 2 others, 1 if burgers-kolmo
code_c=4
init_type={'A':{'type':'orthogonal','gain':1},'B':{'type':'orthogonal','gain':1},'weight':{'type':'orthogonal','gain':1}}
is_complete=False
type_augment=''
regul=False
factor=1

python3 train.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "model.code_c=$code_c" "optim.init_type=$init_type" "model.factor=$factor" "model.is_complete=$is_complete" "optim.regul=$regul" "model.type_augment=$type_augment"