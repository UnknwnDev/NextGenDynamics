# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="SpiderBotAIProject-v0",
    entry_point=f"{__name__}.env:SpiderBotAIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SpiderBotAIEnvCfg",
        "skrl_custom_cfg_entry_point": f"{agents.__name__}:skrl_custom_ppo_cfg.yaml",
    },
)
