#!/bin/bash
#SBATCH --job-name=fuels
#SBATCH --partition=jazzy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/aphinity/%x-%j.out
#SBATCH --error=slurm_run/aphinity/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fuels

dataset_name='pendulum' # pendulum, burgers, gs, lv
batch_size_train=4 # 4 if lv, 16 if pendulum, 1 if gs, 4 if burgers
batch_size_val=32
epochs=1000
lr=0.001
seed=123
hidden_c=64
state_c=2 # 2 others, 1 if burgers-kolmo
type_augment='additive'
is_complete=False
run_name='wobbly-sound-1940'

python3 baselines/APHINITY/aphinity_adapt.py "data.dataset_name=$dataset_name" "optim.batch_size_train=$batch_size_train" "optim.batch_size_val=$batch_size_val" "data.seed=$seed" "optim.epochs=$epochs" "optim.lr=$lr"  "model.hidden_c=$hidden_c" "model.state_c=$state_c" "optim.init_type=$init_type" "model.type_augment=$type_augment" "model.is_complete=$is_complete" "pretrain.run_name=$run_name"


# burgers ethereal-disco-1234
# gs light-tree-1301
# pendulum wobbly-sound-1940