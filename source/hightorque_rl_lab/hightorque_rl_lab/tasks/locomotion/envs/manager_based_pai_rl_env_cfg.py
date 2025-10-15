# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg



@configclass
class ManagerBasedPaiRLEnvCfg(ManagerBasedRLEnvCfg):


    gait: object = MISSING

