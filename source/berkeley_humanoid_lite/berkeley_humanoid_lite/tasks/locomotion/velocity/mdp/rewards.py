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


def track_lin_vel_x_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of forward linear velocity (x-axis only) in the gravity aligned robot frame using exponential kernel.
    
    This only tracks forward motion, encouraging straight-line penguin-like walking.
    """
    asset = env.scene[asset_cfg.name]
    # Get velocity in robot frame (yaw-aligned)
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    # Only track x-direction (forward)
    lin_vel_x_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - vel_yaw[:, 0])
    reward = torch.exp(-lin_vel_x_error / std**2)
    reward = torch.clamp(reward, min=0.0, max=1.0)
    return reward


def penalize_lateral_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize lateral velocity (y-direction) to encourage straight forward motion only.
    
    This strongly discourages any sideways movement, forcing the robot to walk straight forward.
    """
    asset = env.scene[asset_cfg.name]
    # Get velocity in robot frame (yaw-aligned)
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    # Penalize lateral velocity (y-direction)
    lateral_vel = torch.abs(vel_yaw[:, 1])
    # Use exponential penalty - stronger for larger lateral velocities
    penalty = torch.exp(lateral_vel / 0.1) - 1.0
    penalty = torch.clamp(penalty, min=0.0, max=10.0)
    return penalty


def penalize_rotational_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize rotational velocity (yaw) to encourage straight forward motion only.
    
    This strongly discourages any rotation, forcing the robot to walk straight forward.
    """
    asset = env.scene[asset_cfg.name]
    # Get rotational velocity (yaw - z-axis)
    ang_vel_z = torch.abs(asset.data.root_ang_vel_w[:, 2])
    # Use exponential penalty - stronger for larger rotational velocities
    penalty = torch.exp(ang_vel_z / 0.1) - 1.0
    penalty = torch.clamp(penalty, min=0.0, max=10.0)
    return penalty


def com_stability(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward center of mass stability by penalizing large CoM deviations from base.
    
    This encourages the robot to maintain a stable center of mass position relative to the base,
    which is crucial for reliable walking.
    """
    asset = env.scene[asset_cfg.name]
    # Get base position and orientation
    base_quat = asset.data.root_quat_w
    
    # Approximate CoM as base position (for biped, CoM is close to base)
    # Compute CoM velocity in base frame
    com_vel_world = asset.data.root_lin_vel_w[:, :3]
    com_vel_base = quat_rotate_inverse(base_quat, com_vel_world)
    
    # Penalize large lateral and forward CoM velocities (unstable)
    # Reward stability (low CoM velocity in base frame)
    vel_norm = torch.norm(com_vel_base[:, :2], dim=1)
    # Clamp to prevent extreme values
    vel_norm = torch.clamp(vel_norm, min=0.0, max=5.0)
    stability = torch.exp(-vel_norm / 0.5)
    # Ensure no NaN or Inf values
    stability = torch.clamp(stability, min=0.0, max=1.0)
    return stability


def gait_regularity(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.3
) -> torch.Tensor:
    """Reward regular gait patterns by encouraging consistent step timing.
    
    This helps the robot develop a stable, rhythmic walking pattern.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact times for both feet
    contact_times = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    # Compute step frequency (inverse of contact time)
    # Regular gait means consistent timing between steps
    if contact_times.shape[1] >= 2:
        # Difference in contact times between feet (should alternate)
        time_diff = torch.abs(contact_times[:, 0] - contact_times[:, 1])
        # Reward when feet alternate (one in contact, one not)
        alternating = (time_diff > threshold).float()
        # Only reward when moving
        command = env.command_manager.get_command(command_name)
        is_moving = (torch.norm(command[:, :2], dim=1) > 0.1).float()
        reward = alternating * is_moving
        # Ensure no NaN or Inf values
        reward = torch.clamp(reward, min=0.0, max=1.0)
        return reward
    return torch.zeros(env.scene.num_envs, device=env.device)


def recovery_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward successful recovery from perturbations.
    
    This encourages the robot to recover from disturbances and maintain stability.
    """
    asset = env.scene[asset_cfg.name]
    
    # Check if robot is recovering from a fall (orientation improving)
    # Get orientation error (how far from upright)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.scene.num_envs, 1)
    projected_gravity = quat_rotate_inverse(asset.data.root_quat_w, gravity)
    orientation_error = torch.norm(projected_gravity[:, :2], dim=1)
    # Clamp to prevent extreme values
    orientation_error = torch.clamp(orientation_error, min=0.0, max=2.0)
    
    # Reward improvement in orientation (recovery)
    # This is computed as negative orientation error (better orientation = higher reward)
    recovery = torch.exp(-orientation_error / 0.3)
    # Ensure no NaN or Inf values
    recovery = torch.clamp(recovery, min=0.0, max=1.0)
    
    # Bonus for maintaining good orientation after perturbation
    return recovery


def energy_efficiency(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward energy efficiency by penalizing excessive joint velocities and accelerations.
    
    This encourages the robot to walk efficiently with minimal energy consumption.
    Note: We use joint velocities and accelerations as a proxy for energy consumption
    since direct torque access may not be available.
    """
    asset = env.scene[asset_cfg.name]
    
    # Get joint velocities
    joint_velocities = asset.data.joint_vel
    
    # Compute approximate power consumption (velocity squared as proxy)
    # Higher velocities require more energy
    power = torch.sum(torch.square(joint_velocities), dim=1)
    # Clamp to prevent extreme values
    power = torch.clamp(power, min=0.0, max=100.0)
    
    # Reward low power consumption (efficiency)
    efficiency = torch.exp(-power / 5.0)
    # Ensure no NaN or Inf values
    efficiency = torch.clamp(efficiency, min=0.0, max=1.0)
    return efficiency


def com_height_tracking(
    env: ManagerBasedRLEnv, target_height: float = 0.85, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward maintaining a target center of mass height.
    
    This helps the robot maintain consistent body height during walking.
    """
    asset = env.scene[asset_cfg.name]
    
    # Base height (approximate CoM height for biped)
    base_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(base_height - target_height)
    # Clamp to prevent extreme values
    height_error = torch.clamp(height_error, min=0.0, max=2.0)
    
    # Reward maintaining target height
    height_reward = torch.exp(-height_error / 0.1)
    # Ensure no NaN or Inf values
    height_reward = torch.clamp(height_reward, min=0.0, max=1.0)
    return height_reward


def bent_knee_reward(
    env: ManagerBasedRLEnv, target_knee_angle: float = 1.13, asset_cfg: SceneEntityCfg = None
) -> torch.Tensor:
    """Reward maintaining bent knee positions close to default (penguin-like gait).
    
    This encourages the robot to keep knees bent close to the default position,
    similar to how penguins walk. The target matches the default joint position.
    
    Args:
        target_knee_angle: Target knee angle in radians (1.13 rad matches default)
        asset_cfg: SceneEntityCfg specifying which joints to use (knee joints)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"])
    
    asset = env.scene[asset_cfg.name]
    
    # Get knee joint positions - use body_ids which are joint indices
    knee_positions = asset.data.joint_pos[:, asset_cfg.body_ids]
    
    # Reward maintaining knees close to default bent position
    # Target is 1.13 rad (default penguin-like posture)
    knee_error = torch.abs(knee_positions - target_knee_angle)
    knee_error = torch.clamp(knee_error, min=0.0, max=1.0)
    
    # Reward maintaining bent knees close to default (lower error = higher reward)
    knee_reward = torch.exp(-torch.mean(knee_error, dim=1) / 0.2)
    knee_reward = torch.clamp(knee_reward, min=0.0, max=1.0)
    
    return knee_reward


def leg_separation_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None
) -> torch.Tensor:
    """Reward maintaining minimal hip roll and yaw deviation.
    
    This strongly penalizes hip roll and yaw movement to keep legs separated
    and prevent unwanted rotation. Hip roll and yaw should be minimal for
    penguin-like straight walking.
    
    Args:
        asset_cfg: SceneEntityCfg specifying which joints to use (hip roll and yaw joints)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])
    
    asset = env.scene[asset_cfg.name]
    
    # Get hip roll and yaw joint positions - use body_ids which are joint indices
    hip_positions = asset.data.joint_pos[:, asset_cfg.body_ids]
    
    # Target is 0.0 for both hip roll and yaw (minimal movement)
    # Strongly penalize deviation from zero
    hip_deviation = torch.abs(hip_positions)
    hip_deviation = torch.clamp(hip_deviation, min=0.0, max=1.0)
    
    # Reward maintaining minimal hip roll/yaw (lower deviation = higher reward)
    # Use very tight tolerance (0.05) to strongly discourage hip roll/yaw movement
    separation_reward = torch.exp(-torch.mean(hip_deviation, dim=1) / 0.05)
    separation_reward = torch.clamp(separation_reward, min=0.0, max=1.0)
    
    return separation_reward


def prevent_knee_straightening(
    env: ManagerBasedRLEnv, min_knee_angle: float = 0.8, asset_cfg: SceneEntityCfg = None
) -> torch.Tensor:
    """Penalize knee straightening to maintain bent knee posture.
    
    This strongly penalizes when knees straighten below a minimum threshold,
    ensuring the robot maintains bent knees (penguin-like posture) at all times.
    
    Args:
        min_knee_angle: Minimum allowed knee angle in radians (0.8 rad ≈ 45 degrees)
        asset_cfg: SceneEntityCfg specifying which joints to use (knee joints)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"])
    
    asset = env.scene[asset_cfg.name]
    
    # Get knee joint positions - use body_ids which are joint indices
    knee_positions = asset.data.joint_pos[:, asset_cfg.body_ids]
    
    # Penalize when knee angle is below minimum (knee is straightening)
    # Knee pitch should be positive and above threshold for bent knee
    straightening = torch.clamp(min_knee_angle - knee_positions, min=0.0, max=2.0)
    
    # Strong penalty when knees straighten (exponential penalty)
    penalty = torch.exp(torch.mean(straightening, dim=1) / 0.1) - 1.0
    penalty = torch.clamp(penalty, min=0.0, max=10.0)
    
    return penalty


def allow_locomotion_joint_movement(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None, max_deviation: float = 0.3
) -> torch.Tensor:
    """Allow controlled deviation in locomotion joints (hip pitch, knee, ankle pitch).
    
    This allows necessary movement in the main locomotion joints while still
    encouraging them to stay close to default positions. These joints are
    the primary drivers for forward motion.
    
    Args:
        asset_cfg: SceneEntityCfg specifying which joints to use (hip pitch, knee, ankle pitch)
        max_deviation: Maximum allowed deviation from default (radians)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_pitch_joint", ".*_ankle_pitch_joint"])
    
    asset = env.scene[asset_cfg.name]
    
    # Get current and default joint positions
    current_positions = asset.data.joint_pos[:, asset_cfg.body_ids]
    default_positions = asset.data.default_joint_pos[:, asset_cfg.body_ids]
    
    # Compute deviation from default
    deviation = torch.abs(current_positions - default_positions)
    
    # Allow deviation up to max_deviation, then penalize
    # Normalize deviation to [0, 1] range
    normalized_deviation = torch.clamp(deviation / max_deviation, min=0.0, max=1.0)
    
    # Reward staying close to default, but allow some deviation
    # More lenient than hip roll/yaw (allows movement for locomotion)
    reward = torch.exp(-torch.mean(normalized_deviation, dim=1) / 0.3)
    reward = torch.clamp(reward, min=0.0, max=1.0)
    
    return reward


def dual_stance_reward(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward keeping both feet in contact with the ground (dual stance).
    
    This prevents the robot from standing on one leg, encouraging a stable
    penguin-like walking gait where both feet maintain contact most of the time.
    
    Args:
        sensor_cfg: SceneEntityCfg for the contact sensor
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact states for both feet
    contact_times = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_times > 0.01  # Threshold for contact
    
    # Count how many feet are in contact
    num_feet_in_contact = torch.sum(in_contact.int(), dim=1).float()
    
    # Reward when both feet are in contact (dual stance)
    # Maximum reward when both feet are touching
    dual_stance = (num_feet_in_contact >= 2.0).float()
    
    # Bonus: reward maintaining contact (higher contact time = better)
    contact_quality = torch.min(contact_times, dim=1)[0]  # Minimum contact time of both feet
    contact_quality = torch.clamp(contact_quality, min=0.0, max=1.0)
    
    # Combine dual stance reward with contact quality
    reward = dual_stance * (0.5 + 0.5 * contact_quality)
    reward = torch.clamp(reward, min=0.0, max=1.0)
    
    return reward


def maintain_default_joint_positions(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None
) -> torch.Tensor:
    """Reward maintaining all leg joints close to their default positions.
    
    This encourages the robot to stay close to its default posture,
    only making small adjustments (especially hip pitch) for forward motion.
    
    Args:
        asset_cfg: SceneEntityCfg specifying which joints to use (all leg joints)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])
    
    asset = env.scene[asset_cfg.name]
    
    # Get current joint positions and default positions
    current_positions = asset.data.joint_pos[:, asset_cfg.body_ids]
    default_positions = asset.data.default_joint_pos[:, asset_cfg.body_ids]
    
    # Compute deviation from default
    deviation = torch.abs(current_positions - default_positions)
    deviation = torch.clamp(deviation, min=0.0, max=1.0)
    
    # Reward staying close to default (lower deviation = higher reward)
    # Use exponential decay with relaxed threshold to allow necessary movement for locomotion
    # Increased to 0.4 to allow more deviation for forward motion while still encouraging
    # penguin-like posture (bent knees, close to default)
    reward = torch.exp(-torch.mean(deviation, dim=1) / 0.4)
    reward = torch.clamp(reward, min=0.0, max=1.0)
    
    return reward


def slight_hip_pitch_for_forward_motion(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = None, max_deviation: float = 0.1
) -> torch.Tensor:
    """Reward slight hip pitch adjustment for forward motion (penguin-like).
    
    This encourages very small hip pitch changes (close to default) when moving forward,
    similar to how penguins move with minimal hip movement.
    
    Args:
        command_name: Name of the velocity command
        asset_cfg: SceneEntityCfg specifying hip pitch joints
        max_deviation: Maximum allowed deviation from default (radians)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])
    
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get forward velocity command
    forward_vel_cmd = torch.abs(command[:, 0])  # x-direction (forward)
    
    # Get current and default hip pitch positions
    current_hip_pitch = asset.data.joint_pos[:, asset_cfg.body_ids]
    default_hip_pitch = asset.data.default_joint_pos[:, asset_cfg.body_ids]
    
    # Compute deviation from default
    deviation = torch.abs(current_hip_pitch - default_hip_pitch)
    
    # Reward when deviation is small (close to default)
    # Penalize large deviations
    deviation_normalized = torch.clamp(deviation / max_deviation, min=0.0, max=1.0)
    reward = torch.exp(-torch.mean(deviation_normalized, dim=1))
    reward = torch.clamp(reward, min=0.0, max=1.0)
    
    # Only apply when moving forward
    is_moving_forward = (forward_vel_cmd > 0.05).float()
    
    return reward * is_moving_forward
