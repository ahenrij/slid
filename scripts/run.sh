#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1
#SBATCH --account=def-borf
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --ntasks-per-node=1
 
# modules
module load gcc arrow/17.0.0 cuda python/3.10

# python environment
source ~/pyenv2/bin/activate

python src/models/run.py --project veloren --shots 1 --seed $SLURM_ARRAY_TASK_ID
