# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def per_target_time_out(env) -> torch.Tensor:
    return env.command_manager.get_term("waypoint").per_target_timed_out


def died(env) -> torch.Tensor:
    return env.scene.articulations["robot"].data.projected_gravity_b[:, 2] > 0.0


def on_ground(env) -> torch.Tensor:
    return env._base_contact_time > float(env.cfg.base_on_ground_time)
