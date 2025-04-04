#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from app_states.app_state_abc import AppState
from camera_helper import CameraHelper
from utils.gui.gui_input import GuiInput
from utils.gui.text_drawer import TextOnScreenAlignment


class AppStateFreeCamera(AppState):
    def __init__(
        self,
        sandbox_service,
    ):
        self._sandbox_service = sandbox_service
        self._gui_input = self._sandbox_service.gui_input

        config = self._sandbox_service.config
        self._end_on_success = config.habitat.task.end_on_success
        self._success_measure_name = config.habitat.task.success_measure

        self._lookat_pos = None
        self._cam_transform = None

        self._camera_helper = CameraHelper(
            self._sandbox_service.args, self._sandbox_service.gui_input
        )
        self._episode_helper = self._sandbox_service.episode_helper

    def _init_lookat_pos(self):
        random_navigable_point = self.get_sim().sample_navigable_point()
        self._lookat_pos = mn.Vector3(random_navigable_point)

    def _update_lookat_pos(self):
        # update lookat
        move_delta = 0.1
        move = mn.Vector3.zero_init()
        if self._gui_input.get_key(GuiInput.KeyNS.W):
            move.x -= move_delta
        if self._gui_input.get_key(GuiInput.KeyNS.S):
            move.x += move_delta
        if self._gui_input.get_key(GuiInput.KeyNS.O):
            move.y += move_delta
        if self._gui_input.get_key(GuiInput.KeyNS.P):
            move.y -= move_delta
        if self._gui_input.get_key(GuiInput.KeyNS.J):
            move.z += move_delta
        if self._gui_input.get_key(GuiInput.KeyNS.L):
            move.z -= move_delta

        # align move forward direction with lookat direction
        rot_y_rad = -self._camera_helper.lookat_offset_yaw
        rotation = mn.Quaternion.rotation(
            mn.Rad(rot_y_rad),
            mn.Vector3(0, 1, 0),
        )
        self._lookat_pos += rotation.transform_vector(move)

        # draw lookat point
        radius = 0.15
        self._sandbox_service.line_render.draw_circle(
            self._get_camera_lookat_pos(),
            radius,
            mn.Color3(255 / 255, 0 / 255, 0 / 255),
            24,
        )

    @property
    def _env_task_complete(self):
        return (
            self._end_on_success
            and self._sandbox_service.get_metrics()[self._success_measure_name]
        )

    def _env_episode_active(self) -> bool:
        """
        Returns True if current episode is active:
        1) not self._sandbox_service.env.episode_over - none of the constraints is violated, or
        2) not self._env_task_complete - success measure value is not True
        """
        return not (
            self._sandbox_service.env.episode_over or self._env_task_complete
        )

    def _get_camera_lookat_pos(self):
        return self._lookat_pos

    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        if self._episode_helper.next_episode_exists():
            controls_str += "M: next episode\n"

        controls_str += "I, K: look up, down\n"
        controls_str += "A, D: look left, right\n"
        controls_str += "O, P: move up, down\n"
        controls_str += "W, S: move forward, backward\n"

        return controls_str

    def _get_status_text(self):
        status_str = ""

        if not self._env_episode_active():
            if self._env_task_complete:
                status_str += "Task complete!\n"
            else:
                status_str += "Oops! Something went wrong.\n"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len) for line in status_str.split("/n")
        )

        return status_str

    def _update_help_text(self):
        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

        num_episodes_remaining = (
            self._episode_helper.num_iter_episodes
            - self._episode_helper.num_episodes_done
        )
        progress_str = f"{num_episodes_remaining} episodes remaining"
        self._sandbox_service.text_drawer.add_text(
            progress_str,
            TextOnScreenAlignment.TOP_RIGHT,
            text_delta_x=370,
        )

    def get_sim(self):
        return self._sandbox_service.sim

    def on_environment_reset(self, episode_recorder_dict):
        self._init_lookat_pos()
        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

    def sim_update(self, dt, post_sim_update_dict):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._sandbox_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        if (
            self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.M)
            and self._episode_helper.next_episode_exists()
        ):
            self._sandbox_service.end_episode(do_reset=True)

        self._update_lookat_pos()
        if self._env_episode_active():
            self._sandbox_service.compute_action_and_step_env()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()

    def is_app_state_done(self):
        # terminal neverending app state
        return False
