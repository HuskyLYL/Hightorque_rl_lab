import argparse

from isaaclab.app import AppLauncher

from collections import deque

from tqdm import tqdm

import math

import time



import onnxruntime as ort

from isaaclab.utils import configclass

parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

parser.add_argument('--load_model', type=str, required=False,
                        help='Run to load from.',
                        default="./policy/lab/policy_rough.onnx")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

import numpy as np

import torch

import isaaclab.sim as sim_utils


from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import carb.settings
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

from isaaclab.terrains import TerrainImporterCfg

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.terrains.config.rough import *

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import isaaclab.terrains as terrain_gen


class Effort:
    x= 0.
    y= 0.
    z= 0.

class cmd:
    vx = 0.
    vy = 0.
    vyaw = 0.

class gait:
    frequencies = 1.25 # [1.0, 2.5]
    offsets = 0.5 # [0, 1]
    durations = 0.5 # [0.5, 0.5]
    swing_height = 0.05 #  [0.05, 0.1]

class env:
    obs = 42 + 4 + 4 + 3
    num_single_obs = 42 + 4 + 4 + 3
    obs_his = (42 + 4 + 4 + 3) * 10


class Sim2simCfg:
    class sim_config:
        sim_duration = 60.0
        dt = 0.005
        decimation = 5

    class normalization:
        lin_vel=2.0
        ang_vel=0.25
        projected_gravity=1.0
        commands=1.0
        dof_pos=1.0
        dof_vel=0.05
        actions=1.0
        height_scan=5.0

        clip_observations=100.0
        clip_actions=100.0

    class control:
        action_scale = 0.25

    class robot_config:
        kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
        kds = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.double)
        tau_limit = 10.0 * np.ones(12, dtype=np.double)
        use_filter = False




def on_press(key):
    try:
        if key.char == 'w':
            cmd.vx += 0.1
        if key.char == 's':
            cmd.vx -= 0.1
        if key.char == 'a':
            cmd.vyaw += 0.1
        if key.char == 'd':
            cmd.vyaw -= 0.1
        cmd.vx = np.clip(cmd.vx, -0.5, 0.5)
        cmd.vy = np.clip(cmd.vy, -0.5, 0.5)
        cmd.vyaw = np.clip(cmd.vyaw, -0.8, 0.8)

        if key.char == 'j':
            gait.frequencies += 0.2
        if key.char == 'l':
            gait.frequencies -= 0.2
        if key.char == 'i':
            gait.swing_height += 0.1
        if key.char == 'k':
            gait.swing_height -= 0.1
        gait.frequencies = np.clip(gait.frequencies, 1.0, 2.0)
        gait.swing_height = np.clip(gait.swing_height, 0.05, 0.1)


        if key.char == 'r':
            Effort.x += 1
        if key.char == 't':
            Effort.x -= 1
        if key.char == 'f':
            Effort.y += 1
        if key.char == 'g':
            Effort.y -= 1
        if key.char == 'v':
            Effort.z += 1
        if key.char == 'b':
            Effort.z -= 1

        if key.char == '0':
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.vyaw = 0.0
            gait.frequencies = 1.25
            gait.swing_height = 0.05
            Effort.x = 0
            Effort.y = 0
            Effort.z = 0

    except AttributeError:
      pass

def on_release(key):
    if key == keyboard.Key.esc:
        return False
    
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pai_robot_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/isaaclab/isaacsim2sim/usd/pai_v4/pai.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(4., 4., 0.3453),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_thigh_joint": 0.0,
            ".*_calf_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_thigh_joint": 20.0,
                ".*_calf_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 21.0,
                ".*_hip_roll_joint": 21.0,
                ".*_thigh_joint": 21.0,
                ".*_calf_joint": 21.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_thigh_joint": 20.0,
                ".*_calf_joint": 20.0,
            },
            damping={
                ".*_hip_pitch_joint": 0.5,
                ".*_hip_roll_joint": 0.5,
                ".*_thigh_joint": 0.5,
                ".*_calf_joint": 0.5,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 21.1,
                ".*_ankle_roll_joint": 21.1,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.5,
                ".*_ankle_roll_joint": 0.5,
            },
            armature=0.01,
        ),
    },
)

rough_terrains_cfg= TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={

        "A": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1., slope_range=(-0.1, -0.2), platform_width=0.5, border_width=1.0
        ),



   
    },
)





class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

   
    #hf_pyramid_pit_cfg = TerrainGeneratorCfg.hf_pyramid_pit_cfg()
    #ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = pai_robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quaternion_to_euler_array(quat):
        
    # Ensure quaternion is in the correct format [x, y, z, w]

    x, y, z, w = quat
        
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
        
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
        
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
        
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z]) 



def get_obs(robot):

    q = robot.data.joint_pos.cpu().numpy().astype(np.double)[0]

    q_root_link_pos = robot.data.root_link_pos_w.cpu().numpy().astype(np.double)[0]

    q_root_link_quat_w = robot.data.root_link_quat_w.cpu().numpy().astype(np.double)[0]
   
    q_joint_pos = robot.data.joint_pos.cpu().numpy().astype(np.double)[0]

    q = np.concatenate([q_root_link_pos,q_root_link_quat_w,q_joint_pos])

    dq_root = robot.data.root_link_vel_w.cpu().numpy().astype(np.double)[0]

    dq_joint = robot.data.joint_vel.cpu().numpy().astype(np.double)[0]

    dq = np.concatenate([dq_root,dq_joint])

    quat = q_root_link_quat_w[[1,2,3,0]]

    r = R.from_quat(quat)

    v = r.apply(dq[:3], inverse=True).astype(np.double)  # In the base frame

    omega = dq[3:6]

    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    return (q, dq, quat, v, omega, gvec)

 
    


def pd_control(target_q, q, kp, target_dq, dq, kd):

    return (target_q - q) * kp + (target_dq - dq) * kd




def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,policy):

    robot = scene["robot"]
    count_lowlevel = 1
    sim_dt = sim.get_physics_dt()
    sim_time = 0

    

    target_q = np.zeros(12, dtype=np.double)
    last_action = np.zeros(12, dtype=np.double)
    action = np.zeros(12, dtype=np.double)
    hist_obs = np.zeros([1, env.obs_his])
    gait_indices = np.zeros(1, dtype=np.double)
    swing_height_indices = np.zeros(1, dtype=np.double)

 



    

    while True:
    
        q, dq, quat, v, omega, gvec = get_obs(robot)
    


        sim.set_camera_view(q[0:3]+[1,1,1.2],q[0:3])

        q = q[-12:]

        dq = dq[-12:]

        _external_force_b = torch.zeros((robot.num_instances, robot.num_bodies, 3), device="cuda")

        _external_force_b[0][0][0] = Effort.x
        _external_force_b[0][0][1] = Effort.y
        _external_force_b[0][0][2] = Effort.z
        robot.root_physx_view.apply_forces_and_torques_at_position(
            force_data=_external_force_b,
            indices=torch.tensor([0], device=device) ,
            is_global=False,
            position_data=None,
            torque_data=None
        )

        

        if count_lowlevel % Sim2simCfg.sim_config.decimation == 0:
        
            print("\n")

            print("vx:"+str(cmd.vx)+" vy:"+str(cmd.vy)+" vyaw:"+str(cmd.vyaw))

            print("frequencies:"+str(gait.frequencies)+" vy:"+str(gait.durations)+" offsets:"+str(gait.offsets)+" swing_height:"+str(gait.swing_height))

            print("x:"+str(Effort.x)+" y:"+str(Effort.y)+" z:"+str(Effort.z))
            
            print("\n")

            

            obs = np.zeros([1, env.obs])

            _q = quat

            _v = np.array([0.0, 0.0, -1.0])

            projected_gravity = quat_rotate_inverse(_q, _v)

            last_action =action

            obs[0, 0:3] = omega * Sim2simCfg.normalization.ang_vel
            obs[0, 3:6] = projected_gravity
            obs[0, 6:18] = q * Sim2simCfg.normalization.dof_pos
            obs[0, 18:30] = dq * Sim2simCfg.normalization.dof_vel
            obs[0, 30:42] = last_action

            obs[0, 42] = gait.frequencies
            obs[0, 43] = gait.offsets
            obs[0, 44] = gait.durations
            obs[0, 45] = gait.swing_height

            gait_indices = np.remainder(gait_indices + Sim2simCfg.sim_config.dt * Sim2simCfg.sim_config.decimation * gait.frequencies, 1.0)
            swing_height_indices = np.remainder(swing_height_indices + Sim2simCfg.sim_config.dt * Sim2simCfg.sim_config.decimation * gait.frequencies * 2, 1.0)

            desired_contact_states = np.remainder(
                np.hstack(
                    [
                        (gait_indices + gait.offsets + 1),
                        gait_indices
                    ]
                ),
                1.0,
            ) - gait.durations

            first_foot_swing_height_target = gait.swing_height * (np.sin(2 * np.pi * swing_height_indices - 1 / 2 * np.pi) / 2 + 0.5) * (desired_contact_states[0] > 0)
            second_foot_swing_height_target = gait.swing_height * (np.sin(2 * np.pi * swing_height_indices - 1 / 2 * np.pi) / 2 + 0.5) * (desired_contact_states[1] > 0)


            obs[0, 46] = first_foot_swing_height_target
            obs[0, 47] = second_foot_swing_height_target
            obs[0, 48] = gait_indices
            obs[0, 49] = swing_height_indices

            obs[0, 50] = cmd.vx
            obs[0, 51] = cmd.vy
            obs[0, 52] = cmd.vyaw

            hist_obs = np.concatenate((hist_obs[:, env.num_single_obs:], obs[:, :env.num_single_obs]), axis=-1).astype(np.float32)

            input = hist_obs.astype(np.float32)

            action = policy.run(None, {'input': input})[0]

            action = np.clip(
                action,
                -Sim2simCfg.normalization.clip_actions,
                Sim2simCfg.normalization.clip_actions,
            )

            target_q = action * Sim2simCfg.control.action_scale

        target_dq = np.zeros((12), dtype=np.double)


        tau = pd_control(
            target_q, q, Sim2simCfg.robot_config.kps, target_dq, dq, Sim2simCfg.robot_config.kds
        )  # Calc torques
        tau = np.clip(
            tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit
        )[0]  # Clamp torques


        tau = torch.tensor(tau, dtype=torch.float32, device="cuda")

        robot.set_joint_position_target(tau, joint_ids=[0,1,2,3,4,5,6,7,8,9,10,11])



        scene.write_data_to_sim()

        sim.step()

        count_lowlevel +=1

        sim_time += sim_dt
   
        scene.update(sim_dt)




        


def main():

    print(args_cli.device)

    sim2simCfg = Sim2simCfg()

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device,dt=Sim2simCfg.sim_config.dt)

    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([10., 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)

    scene = InteractiveScene(scene_cfg)

    policy = ort.InferenceSession(args_cli.load_model)

    sim.reset()

    print("[INFO]: Setup complete...")
  
    run_simulator(sim, scene,policy)


if __name__ == "__main__":

    main()

    simulation_app.close()



