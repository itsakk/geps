#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/fuels_adapt/%x-%j.out
#SBATCH --error=slurm_run/fuels_adapt/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='combined_big' # pendulum, burgers, gs, lv
batch_size_train=32 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=32
epochs=1000
lr=0.01
seed=123
hidden_c=16
state_c=1 # 2 others, 1 if burgers
code_c=8
is_complete=True
type_augment=''
factor=1 # 1 if lv or pendulum, 0.0005 if gs
run_name='cerulean-tree-2738'

python3 adapt.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "model.code_c=$code_c" "pretrain.run_name=$run_name" "model.factor=$factor" "model.is_complete=$is_complete" "model.type_augment=$type_augment"









#### DATA-DRIVEN
# 'deep-dream-541' = burgers // fearless-lion-1960
# 'resilient-elevator-1046' = gs
# 'misty-grass-974' = kolmo
# 'feasible-glade-1439' = pendulum

### SERIE
# lyric-eon-1035 = kolmo
# lunar-frost-994 = burgers
# old-senate-1450 = pendulum
# earnest-donkey-970 = gs