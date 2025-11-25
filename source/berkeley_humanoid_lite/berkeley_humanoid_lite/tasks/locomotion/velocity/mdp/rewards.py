from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def prevent_feet_collapse(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll"]),
) -> torch.Tensor:
    """Penalize feet being too close together (prevent collapse).
    
    This function strongly penalizes when the left and right feet come too close
    to each other, which is physically impossible in the real world but can
    happen in simulation.
    
    Args:
        env: The RL environment instance
        min_distance: Minimum allowed distance between feet in meters (default: 0.15m)
        asset_cfg: SceneEntityCfg specifying which bodies to use (feet/ankle_roll)
    
    Returns:
        torch.Tensor: Penalty for feet being too close (higher = worse)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get body positions for left and right feet (ankle_roll bodies)
    body_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # Find left and right foot indices
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    left_foot_idx = None
    right_foot_idx = None
    
    for i, name in enumerate(body_names):
        if "left" in name.lower() and "ankle" in name.lower():
            left_foot_idx = i
        elif "right" in name.lower() and "ankle" in name.lower():
            right_foot_idx = i
    
    if left_foot_idx is None or right_foot_idx is None:
        # If we can't find both feet, return zero penalty
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)
    
    # Calculate horizontal distance between feet (XY plane)
    left_foot_pos = body_positions[:, left_foot_idx, :2]  # [num_envs, 2] (x, y)
    right_foot_pos = body_positions[:, right_foot_idx, :2]  # [num_envs, 2] (x, y)
    
    foot_distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)  # [num_envs]
    
    # Penalize when distance is below minimum
    # Use exponential penalty that increases sharply as distance decreases
    distance_deficit = torch.clamp(min_distance - foot_distance, min=0.0)
    penalty = torch.exp(distance_deficit / 0.05) - 1.0  # Exponential penalty
    penalty = torch.clamp(penalty, min=0.0, max=100.0)  # Cap penalty
    
    return penalty


def prevent_knee_collapse(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_knee_.*"]),
) -> torch.Tensor:
    """Penalize knees being too close together (prevent collapse).
    
    This function strongly penalizes when the left and right knees come too close
    to each other, which is physically impossible in the real world but can
    happen in simulation.
    
    Args:
        env: The RL environment instance
        min_distance: Minimum allowed distance between knees in meters (default: 0.12m)
        asset_cfg: SceneEntityCfg specifying which bodies to use (knees)
    
    Returns:
        torch.Tensor: Penalty for knees being too close (higher = worse)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get body positions for left and right knees
    body_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # Find left and right knee indices
    body_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    left_knee_idx = None
    right_knee_idx = None
    
    for i, name in enumerate(body_names):
        if "left" in name.lower() and "knee" in name.lower():
            left_knee_idx = i
        elif "right" in name.lower() and "knee" in name.lower():
            right_knee_idx = i
    
    if left_knee_idx is None or right_knee_idx is None:
        # If we can't find both knees, return zero penalty
        return torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)
    
    # Calculate horizontal distance between knees (XY plane)
    left_knee_pos = body_positions[:, left_knee_idx, :2]  # [num_envs, 2] (x, y)
    right_knee_pos = body_positions[:, right_knee_idx, :2]  # [num_envs, 2] (x, y)
    
    knee_distance = torch.norm(left_knee_pos - right_knee_pos, dim=1)  # [num_envs]
    
    # Penalize when distance is below minimum
    # Use exponential penalty that increases sharply as distance decreases
    distance_deficit = torch.clamp(min_distance - knee_distance, min=0.0)
    penalty = torch.exp(distance_deficit / 0.05) - 1.0  # Exponential penalty
    penalty = torch.clamp(penalty, min=0.0, max=100.0)  # Cap penalty
    
    return penalty


def maintain_hip_yaw_roll_default(
    env: ManagerBasedRLEnv,
    max_deviation: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
) -> torch.Tensor:
    """Penalize deviation of hip yaw and hip roll joints from default positions.
    
    This function penalizes when hip yaw and hip roll joints deviate too much
    from their default positions (0.0), which helps maintain proper leg separation
    and prevents unrealistic gaits.
    
    Args:
        env: The RL environment instance
        max_deviation: Maximum allowed deviation from default in radians (default: 0.1 rad)
        asset_cfg: SceneEntityCfg specifying which joints to use (hip_yaw and hip_roll)
    
    Returns:
        torch.Tensor: Penalty for excessive deviation (higher = worse)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get current joint positions and default positions
    current_positions = asset.data.joint_pos[:, asset_cfg.body_ids]
    default_positions = asset.data.default_joint_pos[:, asset_cfg.body_ids]
    
    # Compute deviation from default
    deviation = torch.abs(current_positions - default_positions)
    
    # Penalize deviations beyond max_deviation
    exceeding_deviation = torch.clamp(deviation - max_deviation, min=0.0)
    
    # Use quadratic penalty for excessive deviations
    penalty = torch.mean(exceeding_deviation**2, dim=1) * 10.0  # Scale penalty
    penalty = torch.clamp(penalty, min=0.0, max=10.0)  # Cap penalty
    
    return penalty
