

import torch

#from typing import TYPE_CHECKING, Literal

#if TYPE_CHECKING:

from isaaclab.envs import ManagerBasedEnv



# 重置步态生成器
def reset_gait_generator(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    env.cfg.gait_generator.resample_gaits(env_ids)
    env.cfg.gait_generator.gait_indices[env_ids] = 0
