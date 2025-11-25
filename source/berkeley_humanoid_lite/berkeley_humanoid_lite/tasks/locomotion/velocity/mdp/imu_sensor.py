"""IMU sensor setup for Isaac Lab using Isaac Sim's IMUSensor API.

Reference: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.sensors.physics/docs/index.html#isaacsim.sensors.physics.IMUSensor
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

try:
    from omni.isaac.sensor import IMUSensor as _IMUSensor
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdGeom
    ISAAC_SIM_IMU_AVAILABLE = True
except ImportError:
    ISAAC_SIM_IMU_AVAILABLE = False
    _IMUSensor = None


def setup_imu_sensors_for_all_envs(env: ManagerBasedRLEnv) -> dict[str, Any]:
    """Setup IMU sensors on robot's base/torso for all environments.
    
    This function creates IMU sensors using Isaac Sim's IMUSensor API
    and attaches them to each robot's base (torso) at the center.
    
    The IMU sensor is placed at the center of the robot's base/torso body.
    
    Args:
        env: The RL environment instance
    
    Returns:
        Dictionary mapping environment indices to IMUSensor instances
    """
    if not ISAAC_SIM_IMU_AVAILABLE:
        print("[WARNING] Isaac Sim IMU sensor API not available. IMU sensors will not be created.")
        return {}
    
    imu_sensors = {}
    
    try:
        # Get the robot asset
        robot = env.scene["robot"]
        
        # Get base body name (typically "base" or root link)
        base_body_names = [name for name in robot.body_names if "base" in name.lower()]
        if not base_body_names:
            # If no "base" body found, use the first body (root link)
            base_body_name = robot.body_names[0] if robot.body_names else None
        else:
            base_body_name = base_body_names[0]
        
        if base_body_name is None:
            print("[ERROR] Could not find robot base body for IMU sensor attachment.")
            return {}
        
        # Get the stage to access prims
        stage = get_current_stage()
        
        # For each environment, create an IMU sensor
        # In Isaac Lab, each environment has its own prim namespace
        num_envs = env.scene.num_envs
        
        # Get the root prim path pattern from robot configuration
        # The robot's root prim path contains the environment namespace
        robot_root_path = robot.root_prim_path
        
        for env_idx in range(num_envs):
            try:
                # Construct prim path for this environment's robot base
                # Replace {ENV_REGEX_NS} with actual environment namespace
                # Isaac Lab typically uses pattern like "/World/envs/env_XXXX/robot"
                if "{ENV_REGEX_NS}" in robot_root_path:
                    # Get actual environment namespace from scene
                    # Isaac Lab creates prims with pattern env_XXXX where XXXX is zero-padded index
                    env_ns = f"env_{env_idx:04d}"
                    base_prim_path = robot_root_path.replace("{ENV_REGEX_NS}", env_ns)
                else:
                    # If no pattern, try to find the actual prim path
                    # Use the robot's body prim paths which should have the correct namespace
                    base_prim_path = f"{robot_root_path}/{base_body_name}"
                
                # Try to get the actual prim path from the stage
                # Check if the base body prim exists
                base_prim = stage.GetPrimAtPath(base_prim_path)
                if not base_prim.IsValid():
                    # Try alternative path construction
                    # Isaac Lab may use different namespace patterns
                    alt_paths = [
                        f"/World/envs/env_{env_idx:04d}/robot/{base_body_name}",
                        f"{robot_root_path}/{base_body_name}",
                    ]
                    for alt_path in alt_paths:
                        alt_prim = stage.GetPrimAtPath(alt_path)
                        if alt_prim.IsValid():
                            base_prim_path = alt_path
                            break
                    else:
                        print(f"[WARNING] Could not find base prim for env {env_idx}, skipping IMU sensor")
                        continue
                
                imu_prim_path = f"{base_prim_path}/IMU"
                
                # Create IMU sensor using Isaac Sim API
                # Reference: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.sensors.physics/docs/index.html#isaacsim.sensors.physics.IMUSensor
                if _IMUSensor is None:
                    raise RuntimeError("IMUSensor class not available")
                imu_sensor = _IMUSensor(
                    prim_path=imu_prim_path,
                    name=f"robot_imu_{env_idx}",
                    translation=(0.0, 0.0, 0.0),  # Center of torso (relative to base)
                    orientation=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
                )
                
                # Initialize and enable the sensor
                imu_sensor.initialize()
                imu_sensor.enable()
                
                imu_sensors[str(env_idx)] = imu_sensor
                
            except Exception as e:
                print(f"[WARNING] Failed to create IMU sensor for env {env_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if imu_sensors:
            print(f"[INFO] Created {len(imu_sensors)} IMU sensors on robot base/torso")
        else:
            print("[WARNING] No IMU sensors were created")
        
        return imu_sensors
        
    except Exception as e:
        print(f"[ERROR] Failed to setup IMU sensors: {e}")
        import traceback
        traceback.print_exc()
        return {}

