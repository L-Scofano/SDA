#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List


class ClientMessageManager:
    r"""
    Extends gfx-replay keyframes to include server messages to be interpreted by the client.
    """
    _message: Dict = {}

    def get_message_dict(self) -> Dict:
        r"""
        Get the server message to be communicated to the client.
        Add a field to this dict to send a message to the client at the end of the frame.
        """
        return self._message

    def clear_message_dict(self) -> None:
        r"""
        Resets the message dict.
        """
        self._message = {}

    def add_highlight(self, pos: List[float], radius: float) -> None:
        r"""
        Draw a highlight circle around the specified position.
        """
        assert pos
        assert len(pos) == 3

        if "highlights" not in self._message:
            self._message["highlights"] = []
        self._message["highlights"].append(
            {"t": [pos[0], pos[1], pos[2]], "r": radius}
        )

    def change_humanoid_position(self, pos: List[float]) -> None:
        r"""
        Change the position of the humanoid.
        Used to synchronize the humanoid position in the client when changing scene.
        """
        self._message["teleportAvatarBasePosition"] = [pos[0], pos[1], pos[2]]

    def signal_scene_change(self) -> None:
        r"""
        Signals the client that the scene is being changed during this frame.
        """
        self._message["sceneChanged"] = True

    def update_navmesh_triangles(self, triangle_vertices):
        r"""
        Send a navmesh. triangle_vertices should be a list of vertices, 3 per triangle.
        Each vertex should be a 3-tuple or similar Iterable of floats.
        """
        assert len(triangle_vertices) > 0
        assert len(triangle_vertices) % 3 == 0
        assert len(triangle_vertices[0]) == 3
        # flatten to a list of floats for more efficient serialization
        self._message["navmeshVertices"] = [
            component for sublist in triangle_vertices for component in sublist
        ]
