"""
Retarget Module

Maps human arm segment angles to robot joint commands with per-joint
gain, offset, and inversion. Applies safety clamping using URDF limits.

This module reconciles human image-plane angles with robot joint axes,
providing explicit configuration for per-joint differences.
"""

import os
import sys
import yaml
import math
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.urdf_loader import load_urdf, URDFLoader  # type: ignore
from vision.arm_geometry import ArmAngles  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class JointMapping:
    """
    Per-joint mapping configuration.
    
    Attributes:
        gain: Scaling factor (human angle * gain = robot angle)
        offset: Offset in degrees to add after scaling
        invert: If True, invert the angle (multiply by -1)
        source_angle: Which human angle to use ('upper_arm', 'forearm', 'shoulder_pitch', 'shoulder_roll', 'elbow')
    """
    gain: float = 1.0
    offset: float = 0.0
    invert: bool = False
    source_angle: str = 'shoulder_pitch'  # 'upper_arm', 'forearm', 'shoulder_pitch', 'shoulder_roll', 'elbow'


class RetargetMapper:
    """
    Maps human arm angles to robot joint commands.
    
    Handles:
    - Per-joint gain, offset, and inversion
    - Safety clamping using URDF limits
    - Mapping from human image-plane angles to robot joint axes
    """
    
    # Right arm joint names in order
    RIGHT_ARM_JOINTS = [
        'arm_right_shoulder_pitch_joint',
        'arm_right_shoulder_roll_joint',
        'arm_right_shoulder_yaw_joint',
        'arm_right_elbow_pitch_joint',
        'arm_right_elbow_roll_joint',
    ]
    
    def __init__(self, urdf_path: str, config_path: Optional[str] = None):
        """
        Initialize retarget mapper.
        
        Args:
            urdf_path: Path to URDF file
            config_path: Path to retarget config YAML (optional)
        """
        # Load URDF for joint limits and info
        self.urdf_loader = load_urdf(urdf_path)
        
        # Load retarget configuration
        if config_path is None:
            config_path = os.path.join(project_root, 'configs', 'retarget.yaml')
        
        self.config_path = config_path
        self.joint_mappings = self._load_retarget_config()
        
        # Get joint names in order
        self.joint_names = self.RIGHT_ARM_JOINTS
        
        logger.info(f"RetargetMapper initialized with {len(self.joint_mappings)} joint mappings")
    
    def _load_retarget_config(self) -> Dict[str, JointMapping]:
        """
        Load retarget configuration from YAML file.
        
        Returns:
            Dictionary mapping joint names to JointMapping objects
        """
        mappings = {}
        
        # Default mappings (will be overridden by config file if it exists)
        default_mappings = {
            'arm_right_shoulder_pitch_joint': JointMapping(gain=1.0, offset=0.0, invert=False, source_angle='shoulder_pitch'),
            'arm_right_shoulder_roll_joint': JointMapping(gain=1.0, offset=0.0, invert=False, source_angle='shoulder_roll'),
            'arm_right_shoulder_yaw_joint': JointMapping(gain=0.0, offset=0.0, invert=False, source_angle='shoulder_roll'),  # Not directly mappable from 2D
            'arm_right_elbow_pitch_joint': JointMapping(gain=1.0, offset=0.0, invert=False, source_angle='elbow'),
            'arm_right_elbow_roll_joint': JointMapping(gain=0.0, offset=0.0, invert=False, source_angle='forearm'),  # Not directly mappable from 2D
        }
        
        # Load from config file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if config and 'joint_mappings' in config:
                    for joint_name, mapping_config in config['joint_mappings'].items():
                        mappings[joint_name] = JointMapping(
                            gain=mapping_config.get('gain', 1.0),
                            offset=mapping_config.get('offset', 0.0),
                            invert=mapping_config.get('invert', False),
                            source_angle=mapping_config.get('source_angle', 'shoulder_pitch')
                        )
                    logger.info(f"Loaded retarget config from {self.config_path}")
                else:
                    mappings = default_mappings
                    logger.warning(f"Config file exists but has no 'joint_mappings', using defaults")
            except Exception as e:
                logger.error(f"Error loading retarget config: {e}, using defaults")
                mappings = default_mappings
        else:
            mappings = default_mappings
            logger.info(f"No retarget config found at {self.config_path}, using defaults")
        
        return mappings
    
    def _get_source_angle(self, arm_angles: Optional[ArmAngles], source_angle: str) -> Optional[float]:
        """
        Get the source angle from ArmAngles based on source_angle type.
        
        Args:
            arm_angles: ArmAngles object or None
            source_angle: Type of angle to extract ('upper_arm', 'forearm', 'shoulder_pitch', 'shoulder_roll', 'elbow')
            
        Returns:
            Angle in degrees or None if not available
        """
        if arm_angles is None:
            logger.debug(f"_get_source_angle: arm_angles is None for source_angle={source_angle}")
            return None
        
        angle_map = {
            'upper_arm': arm_angles.upper_arm_angle,
            'forearm': arm_angles.forearm_angle,
            'shoulder_pitch': arm_angles.shoulder_pitch,
            'shoulder_roll': arm_angles.shoulder_roll,
            'elbow': arm_angles.elbow_angle
        }
        
        angle = angle_map.get(source_angle)
        logger.debug(f"_get_source_angle: source_angle={source_angle}, extracted={angle}°")
        return angle
    
    def map_angle_to_joint(self, joint_name: str, human_angle: Optional[float]) -> Optional[float]:
        """
        Map human angle to robot joint angle using per-joint configuration.
        
        Args:
            joint_name: Name of the robot joint
            human_angle: Human angle in degrees (or None if not available)
            
        Returns:
            Robot joint angle in degrees, or None if input is None
        """
        if human_angle is None:
            logger.debug(f"map_angle_to_joint: {joint_name} - human_angle is None")
            return None
        
        # Get joint mapping configuration
        mapping = self.joint_mappings.get(joint_name)
        if mapping is None:
            logger.warning(f"No mapping found for joint {joint_name}, using default")
            mapping = JointMapping()
        
        logger.debug(f"map_angle_to_joint: {joint_name} - human_angle={human_angle:.2f}°, "
                    f"source_angle={mapping.source_angle}, gain={mapping.gain}, "
                    f"offset={mapping.offset}, invert={mapping.invert}")
        
        # Special handling for elbow angle: 180° (straight) should map to 0° for robot
        # Robot elbow range is -90° to 0° (negative = bent, 0° = straight)
        # Convert: robot_elbow = -(180 - human_elbow) so straight (180°) → 0°, bent (90°) → -90°
        if mapping.source_angle == 'elbow':
            # Elbow angle conversion:
            # Human: 180° = straight, 0° = fully bent
            # Robot: 0° = straight, -90° = fully bent
            # Linear mapping: human_angle 180° → robot 0°, human_angle 0° → robot -90°
            original_angle = human_angle
            # Map from [0, 180] to [-90, 0]
            # Formula: robot = -90 + (90 * human / 180) = -90 + (human / 2)
            human_angle = -90.0 + (human_angle / 2.0)
            logger.debug(f"map_angle_to_joint: {joint_name} - elbow conversion: {original_angle:.2f}° → {human_angle:.2f}°")
        
        # Apply mapping: robot_angle = (human_angle * gain) + offset
        robot_angle = human_angle * mapping.gain
        
        # Apply inversion if needed
        if mapping.invert:
            robot_angle = -robot_angle
        
        # Apply offset
        robot_angle = robot_angle + mapping.offset
        
        logger.debug(f"map_angle_to_joint: {joint_name} - final robot_angle={robot_angle:.2f}°")
        
        return robot_angle
    
    def clamp_to_limits(self, joint_name: str, angle_deg: float) -> float:
        """
        Clamp angle to joint limits from URDF.
        
        Args:
            joint_name: Name of the robot joint
            angle_deg: Angle in degrees to clamp
            
        Returns:
            Clamped angle in degrees
        """
        return self.urdf_loader.clamp_angle_deg(joint_name, angle_deg)
    
    def retarget_arm_angles(self, arm_angles: Optional[ArmAngles]) -> Dict[str, Optional[float]]:
        """
        Retarget human arm angles to robot joint commands for right arm.
        
        Args:
            arm_angles: ArmAngles object for right arm (or None)
            
        Returns:
            Dictionary mapping joint names to target angles in degrees
        """
        joint_commands = {}
        
        # Debug: Log input arm angles
        logger.debug("retarget_arm_angles: Input arm_angles:")
        if arm_angles:
            logger.debug(f"  Right arm: shoulder_pitch={arm_angles.shoulder_pitch:.2f}°, "
                        f"shoulder_roll={arm_angles.shoulder_roll:.2f}°, "
                        f"elbow={arm_angles.elbow_angle:.2f}°, "
                        f"upper_arm={arm_angles.upper_arm_angle:.2f}°, "
                        f"forearm={arm_angles.forearm_angle:.2f}°")
        else:
            logger.debug("  Right arm: None")
        
        # Process right arm joints
        logger.debug(f"retarget_arm_angles: Processing {len(self.RIGHT_ARM_JOINTS)} right arm joints: {self.RIGHT_ARM_JOINTS}")
        
        for joint_name in self.RIGHT_ARM_JOINTS:
            mapping = self.joint_mappings.get(joint_name, JointMapping())
            source_angle = self._get_source_angle(arm_angles, mapping.source_angle)
            robot_angle = self.map_angle_to_joint(joint_name, source_angle)
            
            if robot_angle is not None:
                # Apply safety clamping
                before_clamp = robot_angle
                # Get limits for debugging
                limits = self.urdf_loader.get_joint_limits_deg(joint_name)
                if limits:
                    logger.debug(f"retarget_arm_angles: {joint_name} limits: {limits[0]:.2f}° to {limits[1]:.2f}°")
                robot_angle = self.clamp_to_limits(joint_name, robot_angle)
                if abs(before_clamp - robot_angle) > 0.01:
                    logger.warning(f"retarget_arm_angles: {joint_name} clamped from {before_clamp:.2f}° to {robot_angle:.2f}° (limits: {limits})")
            
            joint_commands[joint_name] = robot_angle
            logger.debug(f"retarget_arm_angles: {joint_name} → {robot_angle}")
        
        logger.debug(f"retarget_arm_angles: Final joint_commands: {joint_commands}")
        return joint_commands
    
    def get_joint_commands(self, arm_angles: Optional[ArmAngles]) -> Dict[str, Optional[float]]:
        """
        Get robot joint commands from human arm angles (alias for retarget_arm_angles).
        
        Args:
            arm_angles: ArmAngles object for right arm (or None)
            
        Returns:
            Dictionary mapping joint names to target angles in degrees
        """
        return self.retarget_arm_angles(arm_angles)
    
    def validate_joint_command(self, joint_name: str, angle_deg: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if a joint command is within limits.
        
        Args:
            joint_name: Name of the robot joint
            angle_deg: Angle in degrees to validate
            
        Returns:
            (is_valid, error_message) tuple
        """
        return self.urdf_loader.validate_angle_deg(joint_name, angle_deg)
    
    def save_retarget_config(self, output_path: Optional[str] = None):
        """
        Save current retarget configuration to YAML file.
        
        Args:
            output_path: Path to save config (defaults to config_path)
        """
        if output_path is None:
            output_path = self.config_path
        
        config = {
            'joint_mappings': {}
        }
        
        for joint_name, mapping in self.joint_mappings.items():
            config['joint_mappings'][joint_name] = {
                'gain': mapping.gain,
                'offset': mapping.offset,
                'invert': mapping.invert,
                'source_angle': mapping.source_angle
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved retarget config to {output_path}")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize retarget mapper
    urdf_path = os.path.join(
        project_root, 
        'source', 
        'berkeley_humanoid_lite_assets', 
        'data', 
        'robots', 
        'berkeley_humanoid', 
        'berkeley_humanoid_lite', 
        'urdf', 
        'berkeley_humanoid_lite.urdf'
    )
    mapper = RetargetMapper(urdf_path)
    
    # Example: Create synthetic arm angles
    from vision.arm_geometry import ArmAngles
    
    test_arm_angles = ArmAngles(
        upper_arm_angle=90.0,
        forearm_angle=90.0,
        shoulder_pitch=0.0,
        shoulder_roll=0.0,
        elbow_angle=180.0
    )
    
    # Retarget to robot joints
    joint_commands = mapper.retarget_arm_angles(test_arm_angles)
    
    print("=" * 60)
    print("Retarget Test Results")
    print("=" * 60)
    for joint_name, angle in joint_commands.items():
        if angle is not None:
            is_valid, error = mapper.validate_joint_command(joint_name, angle)
            status = "✓" if is_valid else "✗"
            print(f"{status} {joint_name}: {angle:.2f}°")
            if not is_valid:
                print(f"    Error: {error}")
        else:
            print(f"✗ {joint_name}: None (no input angle)")
    
    print("=" * 60)
