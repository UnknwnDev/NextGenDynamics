# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotIndices:
    """Cached indices for robot DOFs, bodies, and contact sensors.

    Resolved once after scene setup. Replaces RobotCacheCommandTerm which
    was abusing the CommandTerm pattern for static index storage.
    """

    dof_idx: list[int]
    body_ids: list[int]
    contact_sensor_base_ids: list[int]
    contact_sensor_feet_ids: list[int]
    undesired_contact_body_ids: list[int]

    @classmethod
    def from_scene(cls, scene, cfg) -> RobotIndices:
        robot = scene.articulations["robot"]
        contact_sensor = scene.sensors["contact_sensor"]

        dof_idx, _ = robot.find_joints(cfg.actions.joint_pos.joint_names)
        body_ids, _ = robot.find_bodies(cfg.base_name)
        contact_sensor_base_ids, _ = contact_sensor.find_bodies(cfg.base_name)
        contact_sensor_feet_ids, _ = contact_sensor.find_bodies(cfg.foot_names)
        undesired_contact_body_ids, _ = contact_sensor.find_bodies(cfg.undesired_contact_body_names)

        return cls(
            dof_idx=dof_idx,
            body_ids=body_ids,
            contact_sensor_base_ids=contact_sensor_base_ids,
            contact_sensor_feet_ids=contact_sensor_feet_ids,
            undesired_contact_body_ids=undesired_contact_body_ids,
        )
