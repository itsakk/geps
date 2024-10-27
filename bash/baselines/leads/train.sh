#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=electronic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/leads/%x-%j.out
#SBATCH --error=slurm_run/leads/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='combined' # pendulum, burgers, gs, lv
batch_size_train=16 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=16
epochs=20000
lr=0.001
seed=42
hidden_c=64
state_c=1 # 2 others, 1 if burgers-kolmo
init_type={'weight':{'type':'orthogonal','gain':1}}

python3 baselines/LEADS/leads_train.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "optim.init_type=$init_type"