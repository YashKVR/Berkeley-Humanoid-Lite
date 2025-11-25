"""IMU observation functions for Isaac Lab locomotion environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def imu_linear_acceleration(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get linear acceleration from IMU sensors.
    
    This function reads linear acceleration data from the IMU sensors
    attached to each robot's base/torso.
    
    Args:
        env: The RL environment instance
    
    Returns:
        torch.Tensor: Linear acceleration vectors (num_envs, 3) in m/s²
    """
    if not hasattr(env, "_imu_sensors") or not env._imu_sensors:
        # Fallback: return zeros if IMU sensors are not available
        return torch.zeros((env.scene.num_envs, 3), device=env.device, dtype=torch.float32)
    
    num_envs = env.scene.num_envs
    linear_acc = torch.zeros((num_envs, 3), device=env.device, dtype=torch.float32)
    
    for env_idx in range(num_envs):
        imu_sensor = env._imu_sensors.get(str(env_idx))
        if imu_sensor is not None:
            try:
                # Get current IMU frame data
                imu_data = imu_sensor.get_current_frame()
                if imu_data and "lin_acc" in imu_data:
                    # Convert to tensor (IMU data is typically in numpy array)
                    acc = imu_data["lin_acc"]
                    if isinstance(acc, torch.Tensor):
                        linear_acc[env_idx] = acc
                    else:
                        linear_acc[env_idx] = torch.tensor(acc, device=env.device, dtype=torch.float32)
            except Exception as e:
                # If reading fails, keep zero value
                pass
    
    return linear_acc


def imu_angular_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get angular velocity from IMU sensors.
    
    This function reads angular velocity data from the IMU sensors
    attached to each robot's base/torso.
    
    Args:
        env: The RL environment instance
    
    Returns:
        torch.Tensor: Angular velocity vectors (num_envs, 3) in rad/s
    """
    if not hasattr(env, "_imu_sensors") or not env._imu_sensors:
        # Fallback: return zeros if IMU sensors are not available
        return torch.zeros((env.scene.num_envs, 3), device=env.device, dtype=torch.float32)
    
    num_envs = env.scene.num_envs
    angular_vel = torch.zeros((num_envs, 3), device=env.device, dtype=torch.float32)
    
    for env_idx in range(num_envs):
        imu_sensor = env._imu_sensors.get(str(env_idx))
        if imu_sensor is not None:
            try:
                # Get current IMU frame data
                imu_data = imu_sensor.get_current_frame()
                if imu_data and "ang_vel" in imu_data:
                    # Convert to tensor (IMU data is typically in numpy array)
                    ang_vel = imu_data["ang_vel"]
                    if isinstance(ang_vel, torch.Tensor):
                        angular_vel[env_idx] = ang_vel
                    else:
                        angular_vel[env_idx] = torch.tensor(ang_vel, device=env.device, dtype=torch.float32)
            except Exception as e:
                # If reading fails, keep zero value
                pass
    
    return angular_vel


def imu_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get orientation quaternion from IMU sensors.
    
    This function reads orientation quaternion data from the IMU sensors
    attached to each robot's base/torso.
    
    Args:
        env: The RL environment instance
    
    Returns:
        torch.Tensor: Orientation quaternions (num_envs, 4) in (w, x, y, z) format
    """
    if not hasattr(env, "_imu_sensors") or not env._imu_sensors:
        # Fallback: return identity quaternions if IMU sensors are not available
        quats = torch.zeros((env.scene.num_envs, 4), device=env.device, dtype=torch.float32)
        quats[:, 0] = 1.0  # w component = 1.0 for identity quaternion
        return quats
    
    num_envs = env.scene.num_envs
    orientations = torch.zeros((num_envs, 4), device=env.device, dtype=torch.float32)
    orientations[:, 0] = 1.0  # Initialize with identity quaternions
    
    for env_idx in range(num_envs):
        imu_sensor = env._imu_sensors.get(str(env_idx))
        if imu_sensor is not None:
            try:
                # Get current IMU frame data
                imu_data = imu_sensor.get_current_frame()
                if imu_data and "orientation" in imu_data:
                    # Convert to tensor (IMU data is typically in numpy array)
                    quat = imu_data["orientation"]
                    if isinstance(quat, torch.Tensor):
                        orientations[env_idx] = quat
                    else:
                        orientations[env_idx] = torch.tensor(quat, device=env.device, dtype=torch.float32)
            except Exception as e:
                # If reading fails, keep identity quaternion
                pass
    
    return orientations

