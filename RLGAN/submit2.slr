#!/bin/bash -l
#SBATCH --nodes=1  --time=04:00:00  
#SBATCH -C gpu 
#SBATCH --account m3623
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -c 80

module load pytorch/v1.5.0-gpu

# Start training
srun python train.py ./config.yaml explicit_adv_256_morelate
date

