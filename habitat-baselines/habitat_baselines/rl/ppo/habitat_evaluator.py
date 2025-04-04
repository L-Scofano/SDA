import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import csv
import os


def super_latent_reset(num_environments, out_dim_trajectory):
    super_latent_ = torch.zeros(128, num_environments, out_dim_trajectory)
    return super_latent_


def human_trajectory_reset(num_environments, n_frames_to_consider):
    human_trajectory = torch.zeros(num_environments, n_frames_to_consider, 4)
    return human_trajectory


def super_human_trajectory_reset(num_environments, n_frames_to_consider):
    super_human_trajectory = torch.zeros(
        128, num_environments, n_frames_to_consider, 4
    )
    return super_human_trajectory


def super_history_reset(num_environments, n_frames_to_consider):
    # TODO-REFACTOR : externalize + 2 in config
    super_history = torch.zeros(
        128, num_environments, n_frames_to_consider, 512 + 2
    )
    return super_history


def history_reset(num_environments, n_frames_to_consider):
    # TODO-REFACTOR : externalize + 2 in config
    history = torch.zeros(num_environments, n_frames_to_consider, 512 + 2)
    return history


# TODO-PINLAB : LOG MAGNITUDE
LOG_MAGNITUDE = True


class HabitatEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        self.phase = config.habitat_baselines.phase
        n_environs = config.habitat_baselines.num_environments
        self.n_frames_to_consider = 20

        self.super_latent = super_latent_reset(n_environs, 128)
        self.super_human_trajectory = super_human_trajectory_reset(
            n_environs, self.n_frames_to_consider
        )
        self.super_history = super_history_reset(
            n_environs, self.n_frames_to_consider
        )
        self.human_trajectory = human_trajectory_reset(
            n_environs, self.n_frames_to_consider
        )
        self.history = history_reset(n_environs, self.n_frames_to_consider)

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = (
            {}
        )  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                z = None
                if self.phase == 'first':
                    # Store the human trajectory
                    current_human_localization = batch[
                        "agent_1_localization_sensor"
                    ].unsqueeze(1)
                    self.human_trajectory = self.human_trajectory.to(
                        current_human_localization.device
                    )
                    self.human_trajectory = torch.cat(
                        (
                            self.human_trajectory[:, 1:],
                            current_human_localization,
                        ),
                        dim=1,
                    )

                    # ENCODE INTO A LATENT SPACE
                    human_trajectory_reshaped = self.human_trajectory.view(
                        -1, self.n_frames_to_consider * 4
                    )
                    z = agent.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        latent=None,
                        encode_history_and_return=[
                            "phase1",
                            human_trajectory_reshaped,
                        ],
                        **space_lengths,
                    )
                elif self.phase == "second":
                    self.history = self.history.to(
                        batch["agent_1_localization_sensor"].device
                    )

                    # TODO-PINLAB Here we extract z_hat. We can extract z, however we need the trajectory stored in
                    # encode_history_and_return=['phase1', human_trajectory_reshaped],
                    z, features = agent.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        latent=None,
                        encode_history_and_return=["phase2", self.history],
                        **space_lengths,
                    )

                    # TODO-PINLAB : LOG MAGNITUDE
                    if LOG_MAGNITUDE:
                        current_human_localization = batch[
                            "agent_1_localization_sensor"
                        ].unsqueeze(1)
                        self.human_trajectory = self.human_trajectory.to(
                            current_human_localization.device
                        )
                        self.human_trajectory = torch.cat(
                            (
                                self.human_trajectory[:, 1:],
                                current_human_localization,
                            ),
                            dim=1,
                        )
                        human_trajectory_reshaped = self.human_trajectory.view(
                            -1, self.n_frames_to_consider * 4
                        )
                        z_phase_one = agent.actor_critic.act(
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            latent=None,
                            encode_history_and_return=[
                                "phase1",
                                human_trajectory_reshaped,
                            ],
                            **space_lengths,
                        )

                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    latent=z,
                    deterministic=False,
                    **space_lengths,
                )

                if self.phase == "second":
                    agent0_actions = action_data.actions[:, :2]
                    xt_at = torch.cat(
                        [features, agent0_actions], dim=1
                    ).unsqueeze(1)
                    self.history = torch.cat(
                        (self.history[:, 1:], xt_at), dim=1
                    )

                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            # TODO-PINLAB : here in infos we posses all the informations we need
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            # TODO-PINLAB : here we posses everything to work our magic.
            # current_episodes_info[0] is the scene / environment info for infos[0]
            # from current_episodes_info we can access the log file where z and z_hat have been saved

            # TODO: LOG-MAGNITUDE
            if LOG_MAGNITUDE:
                filename = "z_magnitude_analysis/records.csv"
                csv_keys = [
                        "scene_id",
                        "episode_id",
                        "step",
                        "has_found_human",
                        "human_detected",
                        "steps_after_found_human",
                        "distance_to_target_human",
                        "rotatory_distance_to_target_human",
                        "z_magnitude",
                        "z_hat_magnitude",
                        "z_diff"
                    ]
                directory = os.path.dirname(filename)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    
                if not os.path.exists(filename):
                    with open(filename, "w", newline="", encoding='utf-8') as file:
                        writer = csv.DictWriter(
                            file, fieldnames=csv_keys
                        )
                        writer.writeheader()

                with open(filename, "a", newline="", encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=csv_keys)
                    
                    for i, episode in enumerate(current_episodes_info):
                        
                        scene_id = episode.scene_id
                        scene_id = scene_id.split("data/scene_datasets/hssd-hab/scenes-uncluttered/")[1].split(".scene_instance.json")[0]
                        
                        log_vector = {
                            "scene_id": scene_id,
                            "episode_id": episode.episode_id,
                            "step": infos[i]["num_steps"],
                            "has_found_human": infos[i]["social_nav_stats"][
                                "has_found_human"
                            ],
                            "human_detected": infos[i]["social_nav_stats"][
                                "human_detected"
                            ],
                            "steps_after_found_human": infos[i][
                                "social_nav_stats"
                            ]["follow_human_steps_after_frist_encounter"],
                            "distance_to_target_human": infos[i]["dist_to_goal"],
                            "rotatory_distance_to_target_human": infos[i][
                                "rot_dist_to_goal"
                            ],
                            "z_magnitude": torch.norm(z_phase_one[i]).item(),
                            "z_hat_magnitude": torch.norm(z[i]).item(),
                            "z_diff": torch.norm(z_phase_one[i] - z[i]).item(),
                            
                        }
                        
                        writer.writerow(log_vector)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
