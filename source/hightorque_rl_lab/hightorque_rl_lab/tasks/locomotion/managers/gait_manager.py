import torch

import numpy as np

from collections.abc import Sequence

from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ManagerBase

from isaaclab.utils.dict import class_to_dict

from prettytable import PrettyTable






def torch_rand_float(low, high, size, device='cpu'):

    return (high - low) * torch.rand(size, device=device) + low


class GaitManager:


    def __init__(self, cfg: object, env: ManagerBasedRLEnv):


        self.cfg = cfg

        self.env = env

        # crete buffers to store the command
        self.gaits = torch.zeros(

            self.env.num_envs,

            self.cfg.num_gait_params,

            dtype=torch.float,

            device=self.env.device,

            requires_grad=False,
        )

        self.desired_contact_states = torch.zeros(

            self.env.num_envs,

            2,

            dtype=torch.float,

            device=self.env.device,

            requires_grad=False,
        )

        self.gait_indices = torch.zeros(

            self.env.num_envs, dtype=torch.float, device=self.env.device, requires_grad=False
        )

        self.swing_height_indices = torch.zeros(

            self.env.num_envs, dtype=torch.float, device=self.env.device, requires_grad=False
        )

        self.first_foot_swing_height_target = torch.zeros(

            self.env.num_envs,

            dtype=torch.float,

            device=self.env.device,

            requires_grad=False,
        )

        self.second_foot_swing_height_target = torch.zeros(

            self.env.num_envs,

            dtype=torch.float,

            device=self.env.device,

            requires_grad=False,
        )

        self.gaits_ranges = class_to_dict(self.cfg.ranges)



    def __str__(self) -> str:
        """Returns: A string representation for gait manager configuration."""
        msg = f"<GaitManagerCfg> contains {2 + 4} configuration parameters.\n"

        # create table for main configuration
        table = PrettyTable()
        table.title = "Gait Manager Configuration"
        table.field_names = ["Parameter", "Value", "Type"]
        # set alignment of table columns
        table.align["Parameter"] = "l"
        table.align["Value"] = "r"
        table.align["Type"] = "l"
    
        # add main parameters
        table.add_row(["num_gait_params", self.cfg.num_gait_params, "int"])
        table.add_row(["resampling_time", self.cfg.resampling_time, "float"])
    
        # convert table to string
        msg += table.get_string()
        msg += "\n\n"

        # create table for ranges configuration
        ranges_table = PrettyTable()
        ranges_table.title = "Gait Parameter Ranges"
        ranges_table.field_names = ["Parameter", "Min", "Max", "Range"]
        # set alignment of table columns
        ranges_table.align["Parameter"] = "l"
        ranges_table.align["Min"] = "r"
        ranges_table.align["Max"] = "r"
        ranges_table.align["Range"] = "l"
    
        # add range parameters
        ranges_table.add_row(["frequencies", self.cfg.ranges.frequencies[0], self.cfg.ranges.frequencies[1], f"{self.cfg.ranges.frequencies}"])
        ranges_table.add_row(["offsets", self.cfg.ranges.offsets[0], self.cfg.ranges.offsets[1], f"{self.cfg.ranges.offsets}"])
        ranges_table.add_row(["durations", self.cfg.ranges.durations[0], self.cfg.ranges.durations[1], f"{self.cfg.ranges.durations}"])
        ranges_table.add_row(["swing_height", self.cfg.ranges.swing_height[0], self.cfg.ranges.swing_height[1], f"{self.cfg.ranges.swing_height}"])
    
        # convert ranges table to string
        msg += ranges_table.get_string()
        msg += "\n"

        return msg



    def reset(self, env_ids: Sequence[int] | None):

        extras = {}

        if env_ids == None:

            return
        
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1),
            device=self.env.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1),
            device=self.env.device,
        ).squeeze(1)
        # parts = 4
        # self.gaits[env_ids, 1] = (self.gaits[env_ids, 1] * parts).round() / parts
        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1),
            device=self.env.device,
        ).squeeze(1)
        # parts = 2
        # self.gaits[env_ids, 2] = (self.gaits[env_ids, 2] * parts).round() / parts

        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1),
            device=self.env.device,
        ).squeeze(1)


        extras[f"now_gait{env_ids}:"] = self.gaits[env_ids]

        return extras

    def step_contact_targets(self):
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.env.num_envs, 1),
                self.gaits[:, 2].view(self.env.num_envs, 1),
            ],
            dim=1,
        )
        self.swing_height_indices = torch.remainder(
            self.swing_height_indices + self.env.step_dt * frequencies * 2, 1.0
        )
        self.gait_indices = torch.remainder(
            self.gait_indices + self.env.step_dt * frequencies, 1.0
        )

        self.desired_contact_states = torch.remainder(
            torch.cat(
                [
                    (self.gait_indices + offsets + 1).view(self.env.num_envs, 1),
                    self.gait_indices.view(self.env.num_envs, 1),
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
        
