#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/leads/%x-%j.out
#SBATCH --error=slurm_run/leads/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='pendulum' # pendulum, burgers, gs, lv
batch_size_train=1 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=8
epochs=10000
lr=0.001
seed=123
hidden_c=64
state_c=2 # 2 others, 1 if burgers-kolmo
run_name='stellar-mountain-1946'

python3 baselines/LEADS/leads_adapt.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "pretrain.run_name=$run_name"

# gs 'fearless-bird-1264'
# burgers 'sweet-armadillo-1240'
# kolmo 'efficient-cloud-1311'
# pendulum stellar-mountain-1946