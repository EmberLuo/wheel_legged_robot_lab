# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Implementation of VMC (Virtual Model Control) actions for wheel-legged robots."""

import torch
from typing import Dict, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


@configclass
class VMCActionCfg(ActionTermCfg):
    """Configuration for the VMC action term."""
    
    asset_name: str = "robot"
    """Name of the articulation asset in the scene."""
    
    kp_theta: float = 50.0
    """Proportional gain for leg length control."""
    
    kd_theta: float = 3.0
    """Derivative gain for leg length control."""
    
    kp_l: float = 900.0
    """Proportional gain for leg orientation control."""
    
    kd_l: float = 20.0
    """Derivative gain for leg orientation control."""
    
    action_scale_theta: float = 0.2
    """Scaling factor for theta action."""
    
    action_scale_l: float = 0.1
    """Scaling factor for l action."""
    
    action_scale_vel: float = 20.0
    """Scaling factor for wheel velocity action."""
    
    l_offset: float = 0.25
    """Offset for leg length."""
    
    f_feedforward: float = 100.0
    """Feedforward force."""
    
    leg_params: Dict[str, float] = {"l_thigh": 0.25, "l_shank": 0.3}
    """Leg parameters including thigh and shank lengths."""


class VMCAction(ActionTerm):
    """VMC action term for wheel-legged robots.
    
    This action term implements Virtual Model Control for wheel-legged robots,
    which computes the required torques for the hip and knee joints based on
    virtual leg length and orientation references.
    """
    
    cfg: VMCActionCfg
    _asset: Articulation
    
    def __init__(self, cfg: VMCActionCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)
        
        # extract the robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # get the joint indices for hip and knee joints
        self._hip_indices = self._asset.find_joints(".*_hip")[0]
        self._knee_indices = self._asset.find_joints(".*_knee")[0]
        self._wheel_indices = self._asset.find_joints(".*_wheel_axis")[0]
        
        # create tensors for the actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        # create tensors for VMC computations
        self._theta_hip = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta_knee = torch.zeros(self.num_envs, 2, device=self.device)
        self._l = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta = torch.zeros(self.num_envs, 2, device=self.device)
        self._l_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta_dot = torch.zeros(self.num_envs, 2, device=self.device)
        
        # torques and forces
        self._torque_leg = torch.zeros(self.num_envs, 2, device=self.device)
        self._force_leg = torch.zeros(self.num_envs, 2, device=self.device)
        self._torque_wheel = torch.zeros(self.num_envs, 2, device=self.device)
        self._torque_hip = torch.zeros(self.num_envs, 2, device=self.device)
        self._torque_knee = torch.zeros(self.num_envs, 2, device=self.device)
        
        # pi constant
        self.pi = torch.tensor(3.14159265359, device=self.device)
        
    """
    Properties.
    """
    
    @property
    def action_dim(self) -> int:
        # 6 actions: [theta_left, l_left, wheel_vel_left, theta_right, l_right, wheel_vel_right]
        return 6
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """
    
    def process_actions(self, actions: torch.Tensor):
        """
        process_actions: 继承自ActionTerm，在每个环境步骤（environment step）执行一次，负责预处理发送到环境的原始动作
        """

        # store the raw actions
        self._raw_actions[:] = actions
        # process actions with scaling
        self._processed_actions = torch.zeros_like(actions)
        self._processed_actions[:, 0] = actions[:, 0] * self.cfg.action_scale_theta  # left leg
        self._processed_actions[:, 1] = actions[:, 1] * self.cfg.action_scale_l 
        self._processed_actions[:, 2] = actions[:, 2] * self.cfg.action_scale_vel 
        self._processed_actions[:, 3] = actions[:, 3] * self.cfg.action_scale_theta        # right leg 
        self._processed_actions[:, 4] = actions[:, 4] * self.cfg.action_scale_l  
        self._processed_actions[:, 5] = actions[:, 5] * self.cfg.action_scale_vel    

    def apply_actions(self):
        """
        apply_actions: 继承自ActionTerm，在每个仿真步骤（simulation step）执行一次，负责将处理后的动作应用到资产上
        """

        # compute VMC
        self._compute_vmc()
        
        # initialize torques tensor
        torques = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        
        # assign hip torques
        torques[:, self._hip_indices[0]] = -self._torque_hip[:, 0]  # left hip
        torques[:, self._hip_indices[1]] = -self._torque_hip[:, 1]  # right hip
        
        # assign knee torques
        torques[:, self._knee_indices[0]] = self._torque_knee[:, 0]  # left knee
        torques[:, self._knee_indices[1]] = self._torque_knee[:, 1]  # right knee
        
        # assign wheel torques
        torques[:, self._wheel_indices[0]] = self._torque_wheel[:, 0]  # left wheel
        torques[:, self._wheel_indices[1]] = self._torque_wheel[:, 1]  # right wheel
        
        # apply torques to the robot
        self._asset.set_joint_effort_target(torques)
        
    def _compute_vmc(self):
        """Compute the VMC control law."""
        # Get current joint positions and velocities
        dof_pos = self._asset.data.joint_pos
        dof_vel = self._asset.data.joint_vel
        
        # Leg forward kinematics
        self._theta_hip[:, 0] = -dof_pos[:, self._hip_indices[0]]  # left hip
        self._theta_hip[:, 1] = -dof_pos[:, self._hip_indices[1]]  # right hip
        
        self._theta_knee[:, 0] = dof_pos[:, self._knee_indices[0]] + 0.4396  # left knee
        self._theta_knee[:, 1] = dof_pos[:, self._knee_indices[1]] + 0.4396  # right knee
        
        theta_hip_dot = torch.cat(
            (-dof_vel[:, self._hip_indices[0]].unsqueeze(1), 
             -dof_vel[:, self._hip_indices[1]].unsqueeze(1)), dim=1
        )
        theta_knee_dot = torch.cat(
            (dof_vel[:, self._knee_indices[0]].unsqueeze(1), 
             dof_vel[:, self._knee_indices[1]].unsqueeze(1)), dim=1
        )
        
        # Compute leg forward kinematics
        self._l, self._theta = self._leg_forward_kinematics(
            self._theta_hip, self._theta_knee
        )
        
        # Predict l and theta to calculate l_dot and theta_dot
        dt = 0.001
        l_hat, theta_hat = self._leg_forward_kinematics(
            self._theta_hip + theta_hip_dot * dt, 
            self._theta_knee + theta_knee_dot * dt
        )
        self._l_dot = (l_hat - self._l) / dt
        self._theta_dot = (theta_hat - self._theta) / dt
        
        # Compute references using processed actions
        theta_ref = (
            torch.cat(
                ((self._processed_actions[:, 0]).unsqueeze(1), (self._processed_actions[:, 3]).unsqueeze(1)),
                dim=1,
            )
            * self.cfg.action_scale_theta
        )
        
        l_ref = (
            torch.cat(
                ((self._processed_actions[:, 1]).unsqueeze(1), (self._processed_actions[:, 4]).unsqueeze(1)),
                dim=1,
            )
            * self.cfg.action_scale_l
        ) + self.cfg.l_offset
        
        wheel_speed_ref = (
            torch.cat(
                ((self._processed_actions[:, 2]).unsqueeze(1), (self._processed_actions[:, 5]).unsqueeze(1)),
                dim=1,
            )
            * self.cfg.action_scale_vel
        )
        
        # Compute leg torques and forces
        self._torque_leg = (
            self.cfg.kp_theta * (theta_ref - self._theta)
            - self.cfg.kd_theta * self._theta_dot
        )
        
        self._force_leg = (
            self.cfg.kp_l * (l_ref - self._l)
            - self.cfg.kd_l * self._l_dot
        )
        
        # Compute wheel torques
        self._torque_wheel = 0.5 * (
            wheel_speed_ref - dof_vel[:, self._wheel_indices]
        )
        self._torque_wheel = torch.clip(self._torque_wheel, -5, 5)
        
        # Apply VMC to compute hip and knee torques
        self._torque_hip, self._torque_knee = self._vmc(
            self._force_leg + self.cfg.f_feedforward, self._torque_leg
        )
        
    def _leg_forward_kinematics(self, theta_hip, theta_knee):
        """Compute leg forward kinematics."""
        A = self.cfg.leg_params["l_thigh"]
        B = self.cfg.leg_params["l_shank"]
        
        wheel_x = -A * torch.sin(self.pi - theta_hip) + B * torch.cos(
            -theta_hip + theta_knee + self.pi / 2
        )
        wheel_y = -A * torch.cos(theta_hip) + B * torch.sin(
            -theta_hip + theta_knee + self.pi / 2
        )
        l = torch.sqrt(wheel_x**2 + wheel_y**2)
        theta = torch.arctan2(wheel_x, wheel_y)
        return l, theta
        
    def _vmc(self, F, T):
        """Virtual Model Control computation."""
        A = self.cfg.leg_params["l_thigh"]
        B = self.cfg.leg_params["l_shank"]
        
        theta_shank = -self._theta_hip + self._theta_knee + self.pi / 2
        
        j11 = -A * torch.cos(self._theta_hip) + B * torch.sin(theta_shank)
        j12 = -B * torch.sin(theta_shank)
        j21 = A * torch.sin(self._theta_hip) - B * torch.cos(theta_shank)
        j22 = B * torch.cos(theta_shank)
        
        t_hip = (j11 * torch.sin(self._theta) + j21 * torch.cos(self._theta)) * F + (
            j11 * torch.cos(self._theta) - j21 * torch.sin(self._theta)
        ) / self._l * T
        
        t_knee = (j12 * torch.sin(self._theta) + j22 * torch.cos(self._theta)) * F + (
            j12 * torch.cos(self._theta) - j22 * torch.sin(self._theta)
        ) / self._l * T
        
        return t_hip, t_knee