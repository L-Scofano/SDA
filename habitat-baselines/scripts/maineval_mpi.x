#!/bin/bash

#SBATCH -A IscrC_T2M
#SBATCH -p boost_usr_prod
#SBATCH --time=6:00:00     # format: HH:MM:SS
#SBATCH --nodes=1              # 1 nodes
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval


echo "NODELIST="${SLURM_NODELIST}

export WORLD_SIZE=1
export WANDB_MODE=offline

export MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1`
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf`
echo "MASTER_PORT="$MASTER_PORT

python -u -m habitat_baselines.run \
    --config-name=social_nav/eval.yaml \
    benchmark/multi_agent=hssd_spot_human_social_nav \
    habitat_baselines.evaluate=True \
    habitat_baselines.num_checkpoints=5000 \
    habitat_baselines.total_num_steps=1.0e9 \
    habitat_baselines.num_environments=12 \
    habitat_baselines.video_dir=video_bird_baseline \
    habitat_baselines.checkpoint_folder=checkpoints_social_nav/main_reproduction \
    habitat_baselines.eval_ckpt_path_dir=checkpoints_social_nav/main_reproduction/ckpt.540.pth \
    habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.ang_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True \
    habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.human_stop_and_walk_to_robot_distance_threshold=-1.0 \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.ang_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.ang_speed=10.0 \
    habitat.task.measurements.social_nav_reward.facing_human_reward=3.0 \
    habitat.task.measurements.social_nav_reward.count_coll_pen=0.01 \
    habitat.task.measurements.social_nav_reward.max_count_colls=-1 \
    habitat.task.measurements.social_nav_reward.count_coll_end_pen=5 \
    habitat.task.measurements.social_nav_reward.use_geo_distance=True \
    habitat.task.measurements.social_nav_reward.facing_human_dis=3.0 \
    habitat.task.measurements.social_nav_seek_success.following_step_succ_threshold=400 \
    habitat.task.measurements.social_nav_seek_success.need_to_face_human=True \
    habitat.task.measurements.social_nav_seek_success.use_geo_distance=True \
    habitat.task.measurements.social_nav_seek_success.facing_threshold=0.5 \
    habitat.task.lab_sensors.humanoid_detector_sensor.return_image=True \
    habitat.task.lab_sensors.humanoid_detector_sensor.is_return_image_bbox=True \
    habitat.task.success_reward=10.0 \
    habitat.task.end_on_success=True \
    habitat.task.slack_reward=-0.1 \
    habitat.environment.max_episode_steps=300 \
    habitat.simulator.kinematic_mode=False \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat.simulator.agents.agent_0.joint_start_noise=0.0 \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.test_episode_count=24 \
    habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height=1080 \
    habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width=1920 \
    habitat_baselines.phase=baseline