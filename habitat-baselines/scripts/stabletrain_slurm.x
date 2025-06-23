#!/bin/bash

#SBATCH -A IscrC_T2M
#SBATCH -p boost_usr_prod
#SBATCH --time=12:00:00     # format: HH:MM:SS
#SBATCH --nodes=4            # 4 nodes
#SBATCH --ntasks-per-node=4 # 1 tasks out of 32
#SBATCH --gres=gpu:4       # 4 gpus per node out of 4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=multinode_train_habitat


echo "NODELIST="${SLURM_NODELIST}

export WORLD_SIZE=4
export WANDB_MODE=offline

conda activate habitat

python habitat_baselines/run.py --config-name=social_nav/social_nav.yaml