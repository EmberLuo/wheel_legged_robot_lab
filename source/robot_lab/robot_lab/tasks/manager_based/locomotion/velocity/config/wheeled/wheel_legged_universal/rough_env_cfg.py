import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import math
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.vmc_actions import VMCAction, VMCActionCfg
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    ObservationsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.wheel_legged_universal import WHEEL_LEGGED_UNIVERSAL_CFG  # isort: skip

# use other terrain
ROUGH_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.4,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)

@configclass
class WheelLeggedUniversalActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[".*"], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )
    # vmc = VMCActionCfg(
    #     class_type=VMCAction,
    #     asset_name="robot",
    #     kp_theta = 50.0,  # [N*m/rad]
    #     kd_theta = 3.0,  # [N*m*s/rad]
    #     kp_l = 900.0,  # [N/m]
    #     kd_l = 20.0,  # [N*s/m]
    #     action_scale_theta=0.5,
    #     action_scale_l=0.1,
    #     action_scale_vel=20.0,
    #     l_offset=0.25,
    #     f_feedforward=100.0,
    #     leg_params={"l_thigh": 0.25, "l_shank": 0.3}
    # )


@configclass
class WheelLeggedUniversalRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # Root penalties
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "target_height": 0.35,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"])},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"])},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"])},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"])},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel_axis", "right_wheel_axis"])},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"])},
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"]),
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip", "right_hip", "left_knee", "right_knee"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 100,
            "command_threshold": 0.1,
        },
    )
    base_height = RewTerm(
        func=mdp.base_height,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "target_height": 0.3,
            "max_height_diff": 0.03,
        },
    )
    keep_height = RewTerm(
        func=mdp.keep_height,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "target_height": 0.3,
            "max_height_diff": 0.03,
        },
    )
    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"]),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel_axis", "right_wheel_axis"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["left_hip", "right_hip"],
                ["left_knee", "right_knee"]
            ],
        },
    )

    # Action penalties
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_thigh", "left_shank", "right_thigh", "right_shank", "base_link"]),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"]), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=9.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=6.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # Others
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"]),
            "threshold": 0.5,
        },
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"])},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"]),
            "command_name": "base_velocity"
            },
    )
    wheels_contact_always = RewTerm(
        func=mdp.wheels_contact_always,
        weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel")},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"])},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel", "right_wheel"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wheel", "right_wheel"]),
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wheel", "right_wheel"]),
            "target_height": 0.1,
        },
    )
    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wheel", "right_wheel"]),
            "target_height": -0.2,
        },
    )
    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "synced_feet_pair_names": (("left_wheel", "right_wheel"),),
        },
    )
    upward = RewTerm(
        func=mdp.upward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Termination penalty
    # is_terminated = RewTerm(
    #     func=mdp.is_terminated,
    #     weight=-200.0,
    # )

@configclass
class WheelLeggedUniversalRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    base_link_name = "base_link"
    foot_link_name = ".*_wheel"

    leg_joint_names = [".*_hip", ".*_knee"]
    wheel_joint_names = [".*_wheel_axis"]
    joint_names = [".*"]

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = WHEEL_LEGGED_UNIVERSAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        self.actions = WheelLeggedUniversalActionsCfg()
        self.rewards = WheelLeggedUniversalRewardsCfg()

        # ------------------------------Observations------------------------------
        # Set the joint names for observations

        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # ------------------------------Events------------------------------
        # self.events.randomize_reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-3.14, 3.14),
        #         "pitch": (-3.14, 3.14),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.5, 0.5),
        #         "pitch": (-0.5, 0.5),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.07),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (-0.125, 0.125),
                "y": (-0.125, 0.125),
                "z": (-0.125, 0.125),
                "roll": (-0.125, 0.125),
                "pitch": (-0.125, 0.125),
                "yaw": (-0.125, 0.125),
            },
        }
        
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "WheelLeggedUniversalRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # Disable illegal contact termination to allow wheels to touch ground
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["left_thigh", "left_shank", "right_thigh", "right_shank", "base_link"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        
        # ------------------------------Simulation------------------------------
        # Reduce gravity to help with troubleshooting
        # self.sim.gravity = (0.0, 0.0, -7.0)
