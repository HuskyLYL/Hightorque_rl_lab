from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors import patterns
from isaaclab.managers import EventTermCfg as EventTerm


import math

import hightorque_rl_lab.tasks.locomotion.mdp as mdp





from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from hightorque_rl_lab.assets.pai import Pai_CFG  # isort: skip



# 观测量的标准化处理
@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


# 数据采集的标准化处理
@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


# 机器人的步态生成器
@configclass
class gaitCfg:
    num_gait_params = 4
    resampling_time = 5

    class ranges:
        frequencies = [1.0, 2.0]
        offsets = [0.5, 0.5]
        durations = [0.5, 0.5]
        swing_height = [0.05, 0.1]


@configclass
class PaiRewardCfg:

    track_lin_vel_x_exp = RewTerm(func=mdp.track_lin_vel_x_yaw_frame_exp, weight=1.0, params={"command_name":"base_velocity","std": 0.25})

    track_lin_vel_y_exp = RewTerm(func=mdp.track_lin_vel_y_yaw_frame_exp, weight=2.0, params={"command_name":"base_velocity","std": 0.45})

    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name":"base_velocity","std": 0.5})

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-30.0)

    feet_stumble = RewTerm(
        func=mdp.feet_stumble, 
        weight=-1.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")
        }
    )
    
    feet_regulation = RewTerm(
        func=mdp.feet_regulation, weight=-0.05, 
        params={
            "base_height_target": 0.3454, 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")
        }
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance, 
        weight=-100.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), 
            "min_feet_distance": 0.115
        }
    )

    feet_landing_vel = RewTerm(
        func=mdp.feet_landing_vel, 
        weight=-1.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]), 
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"), 
                "about_landing_threshold": 0.03
        }
    )
    feet_takeoff_vel = RewTerm(
        func=mdp.feet_takeoff_vel, 
        weight=-1.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]), 
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"), 
                "about_takeoff_threshold": 0.03
        }
    )
    tracking_contacts_shaped_force = RewTerm(
        func=mdp.tracking_contacts_shaped_force, 
        weight=-2.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), 
            "gait_force_sigma": 25.0
        }
    )
    tracking_contacts_shaped_vel = RewTerm(
        func=mdp.tracking_contacts_shaped_vel, 
        weight=-2.0, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "gait_vel_sigma": 0.25
        }
    )


    tracking_feet_swing_height= RewTerm(
        func=mdp.tracking_feet_swing_height, 
        weight=5.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "std": 0.03,
        }
    )



@configclass
class PaiSceneCfg(InteractiveSceneCfg):

    # 设置环境的相关信息
    def __init__(self,num_envs,env_spacing,physics_dt,step_dt,enable_height_scan):

        super().__init__(num_envs=num_envs, env_spacing=env_spacing)



        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                curriculum=False,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "plane": terrain_gen.MeshPlaneTerrainCfg()
                },
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            #visual_material=sim_utils.MdlFileCfg(
            #    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            #    project_uvw=True,
            #    texture_scale=(0.25, 0.25),
            #),
            debug_vis=False,
        )

        self.robot: ArticulationCfg = Pai_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt)

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        if enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + "base_link",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6,1.0)),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=(-0.3,0.3)
            )



@configclass
class PaiEventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_bodt_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.1, 5.0),
            "operation": "add",
        },
    )



    # reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # 重置步态生成器
    #reset_gait_generator = EventTerm(func=mdp.reset_gait_generator,mode="reset")

    


    # 这里掉了一个每回合应该进行的操作
    #base_external_force_torque = EventTerm(
    #    func=mdp.apply_external_force_torque,
    #    mode="reset",
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #        "force_range": (0.0, 0.0),
    #        "torque_range": (-0.0, 0.0),
    #    },
    #)





    # interval
    #push_robot = EventTerm(
    #    func=mdp.push_by_setting_velocity,
    #    mode="interval",
    #    interval_range_s=(10.0, 15.0),
    #    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    #)


@configclass
class PaiObservationsCfg:
    """Observation specifications for the MDP."""
    

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        

        # 应该是每个关节的速度
        pai_ang_vel = ObsTerm(
            func=mdp.pai_ang_vel,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_projected_gravity = ObsTerm(
            func=mdp.pai_projected_gravity,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_joint_pos = ObsTerm(
            func=mdp.pai_joint_pos,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_joint_vel = ObsTerm(
            func=mdp.pai_joint_vel,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_action = ObsTerm(
            func=mdp.pai_action,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )
        pai_gait = ObsTerm(
            func=mdp.pai_gait,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        first_foot_swing_height_target = ObsTerm(
            func=mdp.pai_first_foot_swing_height_target,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_second_foot_swing_height_target = ObsTerm(
            func=mdp.pai_second_foot_swing_height_target,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_gait_indices = ObsTerm(
            func=mdp.pai_gait_indices,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_swing_height_indicies = ObsTerm(
            func=mdp.pai_swing_height_indicies,
            params={"obs_scales": NormalizationCfg().obs_scales},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )

        pai_command = ObsTerm(
            func=mdp.pai_command,
            params={"obs_scales": NormalizationCfg().obs_scales,
                    "command_name":"base_velocity"
                    
                    },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )


        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        #projected_gravity = ObsTerm(
        #    func=mdp.projected_gravity,
        #    noise=Unoise(n_min=-0.05, n_max=0.05),
        #)
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        #joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        #joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        #actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    # critical 和model的观测值不一样
    @configclass
    class CriticCfg(PolicyCfg):



        pai_root_lin_vel = ObsTerm(
            func=mdp.pai_root_lin_vel,
            params={"obs_scales": NormalizationCfg().obs_scales,  },
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        pai_feet_contace = ObsTerm(
            func=mdp.pai_feet_contact,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            #clip=(-1.0, 1.0),
        )






    # 这里最终定位到了我的奖励函数
    # policy返回的是obs里面的东西
    policy: PolicyCfg = PolicyCfg()
    critical: CriticCfg = CriticCfg()



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[ ".*_hip_pitch_joint",".*_hip_roll_joint",".*_thigh_joint",".*_calf_joint",".*_ankle_pitch_joint", ".*_ankle_roll_joint"], scale=1.0, use_default_offset=True)








@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["base_link"]), "threshold": 1.0},
    )







@configclass
class PaiFlatEnvCfg(LocomotionVelocityRoughEnvCfg):



    def __post_init__(self):
 
        super().__post_init__()

        self.decimation = 10  

        self.episode_length_s = 20.0

        self.sim.dt = 0.002  # 500 Hz
        self.sim.device = "cuda:0"
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"


        # 完工
        self.scene = PaiSceneCfg(num_envs=4096,env_spacing=2.5,physics_dt=self.sim.dt,step_dt=(self.decimation*self.sim.dt),enable_height_scan=True)


        # 完工
        self.rewards: PaiRewardCfg = PaiRewardCfg()

        # 完工
        self.gait :gaitCfg = gaitCfg()
   
        # 待补全
        self.events: PaiEventCfg = PaiEventCfg()

        # 完成
        self.actions: ActionsCfg = ActionsCfg()

        # 完成
        self.observations:PaiObservationsCfg = PaiObservationsCfg ()


        # 完成
        self.commands: CommandsCfg = CommandsCfg()

        # 超时或者碰撞管理器解除到力
        self.terminations: TerminationsCfg = TerminationsCfg()




        









#class SpotFlatEnvCfg_PLAY(SpotFlatEnvCfg):
#    def __post_init__(self) -> None:
#        # post init of parent
#        super().__post_init__()

        # make a smaller scene for play
#        self.scene.num_envs = 50
#        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
#        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
#        if self.scene.terrain.terrain_generator is not None:
#            self.scene.terrain.terrain_generator.num_rows = 5
#            self.scene.terrain.terrain_generator.num_cols = 5
#            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
#        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
