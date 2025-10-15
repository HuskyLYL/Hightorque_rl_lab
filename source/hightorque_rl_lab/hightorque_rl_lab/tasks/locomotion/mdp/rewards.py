from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.assets import Articulation

from hightorque_rl_lab.tasks.locomotion.envs.manager_based_pai_rl_env import ManagerBasedPaiRLEnv





def track_lin_vel_x_yaw_frame_exp(env:  ManagerBasedPaiRLEnv, command_name :str,std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])

  
    coeffi = (
        env.gait_manager.first_foot_swing_height_target + 
        env.gait_manager.second_foot_swing_height_target
    ) / env.gait_manager.gaits[:, 3] - 0.5
    command = env.command_manager.get_command(command_name)[:, 0] * (1 + 0.4 * coeffi)
    lin_vel_error = torch.square(command - vel_yaw[:, 0])
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_y_yaw_frame_exp(env: ManagerBasedPaiRLEnv, command_name:str,std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    coeffi = 0
    command = env.command_manager.get_command(command_name)[:, 1] * (1 + 0.4 * coeffi)
    lin_vel_error = torch.square(command - vel_yaw[:, 1])
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env: ManagerBasedPaiRLEnv, command_name:str,std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
  
    coeffi = (
        env.gait_manager.first_foot_swing_height_target + 
        env.gait_manager.second_foot_swing_height_target
    ) / env.gait_manager.gaits[:, 3] - 0.5
    command = env.command_manager.get_command(command_name)[:, 2] * (1 + 0.4 * coeffi)
    ang_vel_error = torch.square(command - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)





def joint_acc_l2(env: ManagerBasedPaiRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedPaiRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)







def flat_orientation_l2(env: ManagerBasedPaiRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)








def feet_stumble(
        env: ManagerBasedPaiRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]

    for i in range(2):
        contact_force[:, i] = math_utils.quat_apply_inverse(math_utils.yaw_quat(feet_quat[:, i]), contact_force[:, i])
    
    return torch.any(torch.norm(contact_force[:, :, :2], dim=2) > 5 * torch.abs(contact_force[:, :, 2]), dim=1)


def feet_distance(env:ManagerBasedPaiRLEnv, 
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                  min_feet_distance: float = 0.115
                  ) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]

    terrain_height_scan = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].mean(dim=-1).unsqueeze(1)

    foot_ids = [11, 12]

    feet_pos_f = env.scene["robot"].data.body_pos_w[:, foot_ids, :]
    feet_pos_f[:, :, 2] += -0.048 - terrain_height_scan
    feet_pos_f[:, :, 2] = torch.clip(feet_pos_f[:, :, 2], 0, 1)

    feet_quat = env.scene["robot"].data.body_quat_w[:, foot_ids, :]
    delta_feet_pos = torch.zeros_like(feet_pos_f)
    delta_feet_pos[..., 0] -= 0.05519231
    delta_feet_pos[:, 0] = math_utils.quat_apply(feet_quat[:, 0], delta_feet_pos[:, 0])
    delta_feet_pos[:, 1] = math_utils.quat_apply(feet_quat[:, 1], delta_feet_pos[:, 1])
    feet_pos_h = feet_pos_f + delta_feet_pos
    feet_pos_h[:, :, 2] = torch.clip(feet_pos_h[:, :, 2], 0, 1)




    feet_pos_f_left = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), feet_pos_f[:, 0])
    feet_pos_f_right = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), feet_pos_f[:, 1])
    feet_pos_h_left = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), feet_pos_h[:, 0])
    feet_pos_h_right = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), feet_pos_h[:, 1])

    # feet_distance_f = torch.norm(feet_pos_f[:, 0, :2] - feet_pos_f[:, 1, :2], dim=-1)
    # feet_distance_h = torch.norm(feet_pos_h[:, 0, :2] - feet_pos_h[:, 1, :2], dim=-1)

    # feet_distance = torch.min(feet_distance_f, feet_distance_h)
    # if env.command_generator.cfg.ranges.ang_vel_z[1] == 0:
    #     return torch.clip(min_feet_distance - feet_distance, 2, 1)
    # else:
    #     return torch.clip(min_feet_distance - feet_distance, 0, 1) * (
    #         (env.command_generator.cfg.ranges.ang_vel_z[1] - env.command_generator.command[:, 2]) / 
    #         env.command_generator.cfg.ranges.ang_vel_z[1]
    #     )

    feet_distance_f = torch.abs(feet_pos_f_left[:, 1] - feet_pos_f_right[:, 1])
    feet_distance_h = torch.abs(feet_pos_h_left[:, 1] - feet_pos_h_right[:, 1])
    feet_distance = torch.min(feet_distance_f, feet_distance_h)
    return torch.clip(min_feet_distance - feet_distance, 0, 1)


def feet_regulation(
        env: ManagerBasedPaiRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        base_height_target: float = 0.3454, 
        ) -> torch.Tensor:



    feet_height_target = base_height_target * 0.001
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]


    terrain_height_scan = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].mean(dim=-1).unsqueeze(1)

    foot_ids = [11, 12]

    feet_pos_f = env.scene["robot"].data.body_pos_w[:, foot_ids, :]
    feet_pos_f[:, :, 2] += -0.048 - terrain_height_scan
    feet_pos_f[:, :, 2] = torch.clip(feet_pos_f[:, :, 2], 0, 1)

    feet_quat = env.scene["robot"].data.body_quat_w[:, foot_ids, :]
    delta_feet_pos = torch.zeros_like(feet_pos_f)
    delta_feet_pos[..., 0] -= 0.05519231
    delta_feet_pos[:, 0] = math_utils.quat_apply(feet_quat[:, 0], delta_feet_pos[:, 0])
    delta_feet_pos[:, 1] = math_utils.quat_apply(feet_quat[:, 1], delta_feet_pos[:, 1])
    feet_pos_h = feet_pos_f + delta_feet_pos
    feet_pos_h[:, :, 2] = torch.clip(feet_pos_h[:, :, 2], 0, 1)





    feet_height = (feet_pos_f[:, :, 2] + feet_pos_h[:, :, 2]) / 2

    return torch.sum(
        torch.exp(-feet_height / feet_height_target)
        * torch.square(torch.norm(feet_vel, dim=-1)),
        dim=1,
    )


def feet_landing_vel(env: ManagerBasedPaiRLEnv, 
                    about_landing_threshold: float,
                    sensor_cfg: SceneEntityCfg, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0

    asset: Articulation = env.scene[asset_cfg.name]



    terrain_height_scan = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].mean(dim=-1).unsqueeze(1)

    foot_ids = [11, 12]

    feet_pos_f = env.scene["robot"].data.body_pos_w[:, foot_ids, :]
    feet_pos_f[:, :, 2] += -0.048 - terrain_height_scan
    feet_pos_f[:, :, 2] = torch.clip(feet_pos_f[:, :, 2], 0, 1)

    feet_quat = env.scene["robot"].data.body_quat_w[:, foot_ids, :]
    delta_feet_pos = torch.zeros_like(feet_pos_f)
    delta_feet_pos[..., 0] -= 0.05519231
    delta_feet_pos[:, 0] = math_utils.quat_apply(feet_quat[:, 0], delta_feet_pos[:, 0])
    delta_feet_pos[:, 1] = math_utils.quat_apply(feet_quat[:, 1], delta_feet_pos[:, 1])
    feet_pos_h = feet_pos_f + delta_feet_pos
    feet_pos_h[:, :, 2] = torch.clip(feet_pos_h[:, :, 2], 0, 1)





    feet_height = (feet_pos_f[:, :, 2] + feet_pos_h[:, :, 2]) / 2

    about_to_land = (feet_height < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    return torch.sum(torch.square(landing_z_vels), dim=1)


def feet_takeoff_vel(env: ManagerBasedPaiRLEnv, 
                    about_takeoff_threshold: float,
                    sensor_cfg: SceneEntityCfg, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0

    asset: Articulation = env.scene[asset_cfg.name]


    terrain_height_scan = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].mean(dim=-1).unsqueeze(1)

    foot_ids = [11, 12]

    feet_pos_f = env.scene["robot"].data.body_pos_w[:, foot_ids, :]
    feet_pos_f[:, :, 2] += -0.048 - terrain_height_scan
    feet_pos_f[:, :, 2] = torch.clip(feet_pos_f[:, :, 2], 0, 1)

    feet_quat = env.scene["robot"].data.body_quat_w[:, foot_ids, :]
    delta_feet_pos = torch.zeros_like(feet_pos_f)
    delta_feet_pos[..., 0] -= 0.05519231
    delta_feet_pos[:, 0] = math_utils.quat_apply(feet_quat[:, 0], delta_feet_pos[:, 0])
    delta_feet_pos[:, 1] = math_utils.quat_apply(feet_quat[:, 1], delta_feet_pos[:, 1])
    feet_pos_h = feet_pos_f + delta_feet_pos
    feet_pos_h[:, :, 2] = torch.clip(feet_pos_h[:, :, 2], 0, 1)




    feet_height = (feet_pos_f[:, :, 2] + feet_pos_h[:, :, 2]) / 2
    about_to_land = (feet_height < about_takeoff_threshold) & (~contacts) & (z_vels > 0.0)
    takeoff_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    return torch.sum(torch.square(takeoff_z_vels), dim=1)


def tracking_contacts_shaped_force(
        env: ManagerBasedPaiRLEnv, 
        gait_force_sigma: float,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    
   
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim = -1)
    desired_swing_mask = env.gait_manager.desired_contact_states > 0.0

    reward = 0
    for i in range(2):
        reward += desired_swing_mask[:, i] * (1 - (torch.exp(-foot_forces[:, i] ** 2 / gait_force_sigma)))
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward


def tracking_contacts_shaped_vel(env: ManagerBasedPaiRLEnv, 
                                 gait_vel_sigma: float,
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                 ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    feet_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim = -1)
    desired_contact_mask = env.gait_manager.desired_contact_states < 0.0
    reward = 0
    for i in range(2):
        reward += desired_contact_mask[:, i] * (1 - (torch.exp(-feet_vel[:, i] ** 2 / gait_vel_sigma)))
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward





def tracking_feet_swing_height(env: ManagerBasedPaiRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]


    terrain_height_scan = env.scene.sensors["height_scanner"].data.ray_hits_w[..., 2].mean(dim=-1).unsqueeze(1)

    foot_ids = [11, 12]

    feet_pos_f = env.scene["robot"].data.body_pos_w[:, foot_ids, :]
    feet_pos_f[:, :, 2] += -0.048 - terrain_height_scan
    feet_pos_f[:, :, 2] = torch.clip(feet_pos_f[:, :, 2], 0, 1)

    feet_quat = env.scene["robot"].data.body_quat_w[:, foot_ids, :]
    delta_feet_pos = torch.zeros_like(feet_pos_f)
    delta_feet_pos[..., 0] -= 0.05519231
    delta_feet_pos[:, 0] = math_utils.quat_apply(feet_quat[:, 0], delta_feet_pos[:, 0])
    delta_feet_pos[:, 1] = math_utils.quat_apply(feet_quat[:, 1], delta_feet_pos[:, 1])
    feet_pos_h = feet_pos_f + delta_feet_pos
    feet_pos_h[:, :, 2] = torch.clip(feet_pos_h[:, :, 2], 0, 1)

    feet_height_f = feet_pos_f[:, :, 2]
    feet_height_h = feet_pos_h[:, :, 2]

   

    feet_height_target_left = env.gait_manager.first_foot_swing_height_target
    feet_height_target_right = env.gait_manager.second_foot_swing_height_target

    feet_height_error_left_f = torch.abs((feet_height_target_left - feet_height_f[:, 0]))
    feet_height_error_left_h = torch.abs((feet_height_target_left - feet_height_h[:, 0]))
    feet_height_error_right_f = torch.abs((feet_height_target_right - feet_height_f[:, 1]))
    feet_height_error_right_h = torch.abs((feet_height_target_right - feet_height_h[:, 1]))

    # feet_height_error_left = torch.square(torch.cat([feet_height_error_left_f.unsqueeze(-1), feet_height_error_left_h.unsqueeze(-1)], dim=-1)).mean(-1)
    # feet_height_error_right = torch.square(torch.cat([feet_height_error_right_f.unsqueeze(-1), feet_height_error_right_h.unsqueeze(-1)], dim=-1)).mean(-1)
    feet_height_error_left = torch.square(feet_height_error_left_h)
    feet_height_error_right = torch.square(feet_height_error_right_h)

    left_feet_swing_mask = env.gait_manager.desired_contact_states[:, 0] > 0
    right_feet_swing_mask = env.gait_manager.desired_contact_states[:, 1] > 0
    reward = (
        torch.exp(-feet_height_error_left / std**2)*left_feet_swing_mask + 
        torch.exp(-feet_height_error_right / std**2)*right_feet_swing_mask
    ) / 2
    # no reward for zero command
    # reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    return reward



