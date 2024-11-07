#!/bin/bash
#SBATCH --job-name=xxx
#SBATCH --partition=xxx
#SBATCH --constraint xxx
#SBATCH --nodes=xxx
#SBATCH --gpus-per-node=xxx
#SBATCH --time=xxx
#SBATCH --output=xxx
#SBATCH --error=xxx

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate geps

dataset_name='xxx' # pendulum, burgers, gs, lv, combined, gs_multi
batch_size_train=128 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=128
epochs=20000
lr=0.01
seed=42
hidden_c=64
state_c=1
code_c=8
is_complete=True
type_augment=''
regul=False
factor=1
init_type={'A':{'type':'orthogonal','gain':1},'B':{'type':'orthogonal','gain':1},'weight':{'type':'orthogonal','gain':1}}

python3 train.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "model.code_c=$code_c" "optim.init_type=$init_type" "model.factor=$factor" "model.is_complete=$is_complete" "optim.regul=$regul" "model.type_augment=$type_augment"
