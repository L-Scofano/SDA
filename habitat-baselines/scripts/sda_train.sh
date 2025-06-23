#!/bin/bash

# sda phase, select in [baseline, first, second]
phase=second

# specify checkpoint folder, where the checkpoints will be saved.
checkpoint_folder=YOUR_CHECKPOINT_FOLDER_PATH

# not used during training. 
eval_checkpoint_path=YOUR_EVAL_CHECKPOINT_PATH

python -u -m habitat_baselines.run \
    --config-name=social_nav/social_nav_working.yaml \
    benchmark/multi_agent=hssd_spot_human_social_nav_working \
    habitat_baselines.evaluate=False \
    habitat_baselines.num_checkpoints=5000 \
    habitat_baselines.total_num_steps=1.0e9 \
    habitat_baselines.num_environments=24 \
    habitat_baselines.tensorboard_dir=tb_social_nav \
    habitat_baselines.video_dir=video_social_nav \
    habitat_baselines.checkpoint_folder=$checkpoint_folder \
    habitat_baselines.eval_ckpt_path_dir=$eval_checkpoint_path \
    habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.ang_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True \
    habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False \
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
    habitat.environment.max_episode_steps=1500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat.simulator.agents.agent_0.joint_start_noise=0.0 \
    habitat_baselines.phase=$phase \