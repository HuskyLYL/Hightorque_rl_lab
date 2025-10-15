# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym



##
# Register Gym environments.
##
# 小派机器人正常行走的环境
gym.register(
    id="Isaac-Velocity-Flat-Hightorque-pai-v0",
    entry_point="hightorque_rl_lab.tasks.locomotion.envs.manager_based_pai_rl_env:ManagerBasedPaiRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PaiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"hightorque_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

