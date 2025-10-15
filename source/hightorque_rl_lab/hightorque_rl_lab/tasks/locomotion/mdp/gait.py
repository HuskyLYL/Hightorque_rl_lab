import torch

import numpy as np

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedEnvCfg

from isaaclab.utils.dict import class_to_dict



def torch_rand_float(low, high, size, device='cpu'):
    return (high - low) * torch.rand(size, device=device) + low



# 这里要废弃掉，具体的功能要写入具体的奖励函数里面
class Gait:

    def __init__(self, cfg, env_cfg: ManagerBasedEnvCfg):

        #gaitCFG
        self.cfg = cfg

        self.num_envs = env_cfg.scene.num_envs

        self.device = env_cfg.sim.device

        self.dt =env_cfg.sim.dt

        self.decimation = env_cfg.decimation

        # 频率 相位 持续时间 摆动高度 
        self.gaits = torch.zeros(
            self.num_envs,
            self.cfg.num_gait_params,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # 抬起的状态
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.swing_height_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.first_foot_swing_height_target = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.second_foot_swing_height_target = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.gaits_ranges = class_to_dict(self.cfg.ranges)

    def resample_gaits(self, env_ids: Sequence[int] | None):
        if env_ids == None:
            return
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 4
        # self.gaits[env_ids, 1] = (self.gaits[env_ids, 1] * parts).round() / parts
        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 2
        # self.gaits[env_ids, 2] = (self.gaits[env_ids, 2] * parts).round() / parts

        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)



    def update_contact_targets(self,step_id):
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.num_envs, 1),
                self.gaits[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )
        self.swing_height_indices = torch.remainder(
            self.decimation*step_id*self.dt * frequencies * 2, 1.0
        )
        self.gait_indices = torch.remainder(
            self.decimation*step_id*self.dt * frequencies, 1.0
        )

        self.desired_contact_states = torch.remainder(
            torch.cat(
                [
                    (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                    self.gait_indices.view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        ) - durations
        # stance_idxs = self.desired_contact_states < 0.0
        # swing_idxs = self.desired_contact_states > 0.0

        self.first_foot_swing_height_target = self.gaits[:, 3] * (torch.sin(2 * np.pi * self.swing_height_indices - 1 / 2 * np.pi) / 2 + 0.5) * (self.desired_contact_states[:, 0] > 0)
        self.second_foot_swing_height_target = self.gaits[:, 3] * (torch.sin(2 * np.pi * self.swing_height_indices - 1 / 2 * np.pi) / 2 + 0.5) * (self.desired_contact_states[:, 1] > 0)
        
        # print(
        #     self.desired_contact_states[0], 
        #     self.clock_inputs_sin[0]*self.desired_contact_states[0, 0], 
        #     self.clock_inputs_sin[0]*self.desired_contact_states[0, 1]
        # )


