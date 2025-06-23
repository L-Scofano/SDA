#!/bin/bash

#SBATCH -A IscrC_T2M
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00     # format: HH:MM:SS
#SBATCH --nodes=1              # 1 nodes
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=stable_eval

echo "NODELIST="${SLURM_NODELIST}

export WORLD_SIZE=1
export WANDB_MODE=offline

export MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1`
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf`
echo "MASTER_PORT="$MASTER_PORT

python habitat_baselines/run.py --config-name=social_nav/eval.yaml habitat_baselines.evaluate=True habitat_baselines.eval_ckpt_path_dir=./data/checkpoints/ckpt.153.pth habitat_baselines.eval.should_load_ckpt=True