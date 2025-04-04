#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from utils.hablab_utils import get_agent_art_obj_transform

import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index


class GuiNavigationHelper:
    def __init__(self, gui_service, agent_idx):
        self._sandbox_service = gui_service
        self._agent_idx = agent_idx
        self._largest_island_idx = None

    def _get_sim(self):
        return self._sandbox_service.sim

    def _draw_nav_hint_from_agent(
        self, forward_dir, end_pos, end_radius, color
    ):
        assert forward_dir
        agent_idx = self._agent_idx
        assert agent_idx is not None
        art_obj = (
            self._get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        agent_pos = art_obj.transformation.translation

        # if not forward_dir:
        #     transformation = self.cam_transform or art_obj.transformation
        #     forward_dir = transformation.transform_vector(-mn.Vector3(0, 0, 1))
        #     forward_dir[1] = 0
        #     forward_dir = forward_dir.normalized()

        self._draw_nav_hint(
            agent_pos,
            forward_dir,
            end_pos,
            end_radius,
            color,
            self._sandbox_service.get_anim_fraction(),
        )

    def on_environment_reset(self):
        sim = self._get_sim()
        # recompute the largest indoor island id whenever the sim backend may have changed
        self._largest_island_idx = get_largest_island_index(
            sim.pathfinder, sim, allow_outdoor=False
        )

    def _get_humanoid_walk_path_to(self, target_pos):
        agent_root = get_agent_art_obj_transform(
            self._get_sim(), self._agent_idx
        )

        pathfinder = self._get_sim().pathfinder
        # snap target to the selected island
        assert self._largest_island_idx is not None
        snapped_pos = pathfinder.snap_point(
            target_pos, island_index=self._largest_island_idx
        )
        snapped_start_pos = agent_root.translation
        snapped_start_pos.y = snapped_pos.y

        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start_pos
        path.requested_end = snapped_pos
        found_path = pathfinder.find_path(path)

        return found_path, path

    def _get_humanoid_walk_dir_from_path(self, path):
        assert len(path.points) >= 2
        walk_dir = mn.Vector3(path.points[1]) - mn.Vector3(path.points[0])
        return walk_dir

    def _viz_humanoid_walk_path(self, path):
        path_color = mn.Color3(0, 153 / 255, 255 / 255)
        path_endpoint_radius = 0.12

        path_points = []
        for path_i in range(0, len(path.points)):
            adjusted_point = mn.Vector3(path.points[path_i])
            # first point in path is at wrong height
            if path_i == 0:
                adjusted_point.y = mn.Vector3(path.points[path_i + 1]).y
            path_points.append(adjusted_point)

        self._sandbox_service.line_render.draw_path_with_endpoint_circles(
            path_points, path_endpoint_radius, path_color
        )

    def get_humanoid_walk_hints_from_remote_gui_input(
        self, visualize_path=True
    ):
        walk_dir = None
        distance_multiplier = 1.0

        (
            target_pos,
            target_rot_quat,
        ) = self._sandbox_service.remote_gui_input.get_head_pose()

        walk_dir, distance_multiplier = self._get_humanoid_walk_hints(
            target_pos=target_pos,
            target_rot_quat=target_rot_quat,
            visualize_path=visualize_path,
        )

        return walk_dir, distance_multiplier

    def get_humanoid_walk_hints_from_ray_cast(self, visualize_path=True):
        walk_dir = None
        distance_multiplier = 1.0

        target_on_floor = self._get_target_pos_from_ray_cast()
        if target_on_floor is None:
            return walk_dir, distance_multiplier

        walk_dir, distance_multiplier = self._get_humanoid_walk_hints(
            target_pos=target_on_floor,
            target_rot_quat=None,  # habitat_sim.utils.common.random_quaternion() can be used to generate random rotations for testing
            visualize_path=visualize_path,
        )

        return walk_dir, distance_multiplier

    def _get_humanoid_walk_hints(
        self, target_pos, target_rot_quat, visualize_path=True
    ):
        walk_dir = None
        distance_multiplier = 1.0
        geodesic_dist_threshold = 0.05

        found_path, path = self._get_humanoid_walk_path_to(target_pos)
        if (
            found_path
            and len(path.points) >= 2
            and path.geodesic_distance >= geodesic_dist_threshold
        ):
            walk_dir = self._get_humanoid_walk_dir_from_path(path)
            distance_multiplier = 1.0
            if visualize_path:
                self._viz_humanoid_walk_path(path)

        if walk_dir is None and target_rot_quat is not None:
            walk_dir = self._compute_forward_dir(target_rot_quat)
            distance_multiplier = 0.0

        return walk_dir, distance_multiplier

    def _get_target_pos_from_ray_cast(self):
        ray = self._sandbox_service.gui_input.mouse_ray

        floor_y = 0.15  # hardcoded to ReplicaCAD

        if not ray or ray.direction.y >= 0 or ray.origin.y <= floor_y:
            return None

        dist_to_floor_y = (ray.origin.y - floor_y) / -ray.direction.y
        target_on_floor = ray.origin + ray.direction * dist_to_floor_y

        return target_on_floor

    def _compute_forward_dir(self, target_rot_quat):
        direction_vector = mn.Vector3(1.0, 0.0, 0.0)
        heading_vector = target_rot_quat.transform_vector(direction_vector)
        heading_vector.y = 0
        heading_vector = heading_vector.normalized()

        return heading_vector

    @staticmethod
    def _evaluate_cubic_bezier(ctrl_pts, t):
        assert len(ctrl_pts) == 4
        weights = (
            pow(1 - t, 3),
            3 * t * pow(1 - t, 2),
            3 * pow(t, 2) * (1 - t),
            pow(t, 3),
        )

        result = weights[0] * ctrl_pts[0]
        for i in range(1, 4):
            result += weights[i] * ctrl_pts[i]

        return result

    def _draw_nav_hint(
        self, start_pos, start_dir, end_pos, end_radius, color, anim_fraction
    ):
        assert isinstance(start_pos, mn.Vector3)
        assert isinstance(start_dir, mn.Vector3)
        assert isinstance(end_pos, mn.Vector3)

        bias_weight = 0.5
        biased_dir = (
            start_dir + (end_pos - start_pos).normalized() * bias_weight
        ).normalized()

        start_dir_weight = min(4.0, (end_pos - start_pos).length() / 2)
        ctrl_pts = [
            start_pos,
            start_pos + biased_dir * start_dir_weight,
            end_pos,
            end_pos,
        ]

        steps_per_meter = 10
        pad_meters = 1.0
        alpha_ramp_dist = 1.0
        num_steps = max(
            2,
            int(
                ((end_pos - start_pos).length() + pad_meters) * steps_per_meter
            ),
        )

        prev_pos = None
        for step_idx in range(num_steps):
            t = step_idx / (num_steps - 1) + anim_fraction * (
                1 / (num_steps - 1)
            )
            pos = self._evaluate_cubic_bezier(ctrl_pts, t)

            if (pos - end_pos).length() < end_radius:
                break

            if step_idx > 0:
                alpha = min(1.0, (pos - start_pos).length() / alpha_ramp_dist)

                radius = 0.05
                num_segments = 12
                # todo: use safe_normalize
                normal = (pos - prev_pos).normalized()
                color_with_alpha = mn.Color4(color)
                color_with_alpha[3] *= alpha
                self._sandbox_service.line_render.draw_circle(
                    pos, radius, color_with_alpha, num_segments, normal
                )
            prev_pos = pos
