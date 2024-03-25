# 1. Code installation and setup
## pdegen installation
```
conda create -n fuels python=3.11.0
pip install -e .
```

## setup wandb config example

add to your `~/.bashrc`
```
export WANDB_API_TOKEN=your_key
export WANDB_DIR=your_dir
export WANDB_CACHE_DIR=your_cache_dir
export WANDB_CONFIG_DIR="${WANDB_DIR}config/wandb"
export MINICONDA_PATH=your_anaconda_path
```
