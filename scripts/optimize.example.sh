#!/bin/bash -l
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1
#SBATCH --account=def-borf
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --ntasks-per-node=1
 
# modules
module load gcc arrow/17.0.0 cuda python/3.10

export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=/home/haidasso/scratch/slid/src:$PYTHONPATH
export HF_TOKEN="xxx"

# python environment
source ~/slidenv/bin/activate

python src/hf_login.py

python src/slid/modeling/run.py\
    --project inkscape\
    --input-dataset data/inkscape.csv\
    --num-shots 12\
    --output-dir results\
    --model-name BAAI/bge-small-en-v1.5\
    --seed $SLURM_ARRAY_TASK_ID
