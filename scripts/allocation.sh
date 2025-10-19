#! /bin/bash

salloc --time=03:00:00 --account=def-borf --gpus-per-node=v100l:1 --mem=32G --cpus-per-task=1

# modules
module load gcc arrow/17.0.0 cuda python/3.10

# load python env
source ~/pyenv/bin/activate

python src/models/run.py --seed 42
