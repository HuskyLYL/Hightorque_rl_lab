# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera


from hightorque_rl_lab.tasks.locomotion.envs.manager_based_pai_rl_env import ManagerBasedPaiRLEnv


"""
小派观测量
"""


def pai_ang_vel(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b*obs_scales.ang_vel


def pai_projected_gravity(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.projected_gravity_b*obs_scales.projected_gravity

def pai_joint_pos(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    return (asset.data.joint_pos - asset.data.default_joint_pos)*obs_scales.joint_pos

def pai_joint_vel(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    return (asset.data.joint_vel - asset.data.default_joint_vel)*obs_scales.joint_vel


def pai_action(env:  ManagerBasedPaiRLEnv, obs_scales,action_name: str | None = None) -> torch.Tensor:

    if action_name is None:

        #print( env.action_manager.action*obs_scales.actions)

        return env.action_manager.action*obs_scales.actions
    
    else:

        return env.action_manager.get_term(action_name).raw_actions*obs_scales.actions
    
    
def pai_gait(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    return env.gait_manager.gaits

def pai_first_foot_swing_height_target(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    return env.gait_manager.first_foot_swing_height_target.view(env.num_envs,1)

def pai_second_foot_swing_height_target(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    return env.gait_manager.second_foot_swing_height_target.view(env.num_envs,1)

def pai_gait_indices(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    return env.gait_manager.gait_indices.view(env.num_envs,1)

def pai_swing_height_indicies(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    return env.gait_manager.swing_height_indices.view(env.num_envs,1)


def pai_command(env:  ManagerBasedPaiRLEnv, obs_scales,command_name,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    return env.command_manager.get_command(command_name)*obs_scales.commands


def pai_root_lin_vel(env:  ManagerBasedPaiRLEnv, obs_scales,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_lin_vel_b*obs_scales.lin_vel


def pai_feet_contact(env:  ManagerBasedPaiRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")) -> torch.Tensor:

    net_contace_force = env.scene.sensors["contact_sensor"].data.net_forces_w_history

    contact =torch.max(torch.norm(net_contace_force[:, :, asset_cfg.body_ids], dim=-1), dim=1)[0] > 0.5


    return torch.as_tensor(contact, dtype=torch.float32)






