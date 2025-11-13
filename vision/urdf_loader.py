"""
Simple URDF Loader for joint limits and information.

Parses URDF files to extract joint limits and provides clamping functionality.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class URDFLoader:
    """
    Simple URDF loader for extracting joint information.
    """
    
    def __init__(self, urdf_path: str):
        """
        Initialize URDF loader.
        
        Args:
            urdf_path: Path to URDF file
        """
        self.urdf_path = urdf_path
        self.joint_limits = {}
        self._parse_urdf()
    
    def _parse_urdf(self):
        """Parse URDF file to extract joint limits."""
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            
            # Find all joints
            for joint in root.findall('.//joint'):
                joint_name = joint.get('name')
                if joint_name is None:
                    continue
                
                # Find limit element
                limit_elem = joint.find('limit')
                if limit_elem is not None:
                    lower = limit_elem.get('lower')
                    upper = limit_elem.get('upper')
                    
                    if lower is not None and upper is not None:
                        try:
                            lower_rad = float(lower)
                            upper_rad = float(upper)
                            self.joint_limits[joint_name] = (lower_rad, upper_rad)
                            logger.debug(f"Found limits for {joint_name}: {math.degrees(lower_rad):.2f}° to {math.degrees(upper_rad):.2f}°")
                        except ValueError:
                            logger.warning(f"Could not parse limits for {joint_name}: lower={lower}, upper={upper}")
            
            logger.info(f"Loaded limits for {len(self.joint_limits)} joints from URDF")
            
        except Exception as e:
            logger.error(f"Error parsing URDF: {e}")
            # Use default limits if parsing fails
            self._set_default_limits()
    
    def _set_default_limits(self):
        """Set default joint limits for right arm (in radians)."""
        # Default limits from URDF (in radians)
        self.joint_limits = {
            'arm_right_shoulder_pitch_joint': (-0.785398, 1.5708),  # -45° to 90°
            'arm_right_shoulder_roll_joint': (-1.309, 0.261799),   # -75° to 15°
            'arm_right_shoulder_yaw_joint': (-0.785398, 0.785398), # -45° to 45°
            'arm_right_elbow_pitch_joint': (-1.5708, 0.0),        # -90° to 0°
            'arm_right_elbow_roll_joint': (-0.785398, 0.785398),   # -45° to 45°
        }
        logger.info("Using default joint limits")
    
    def get_joint_limits_deg(self, joint_name: str) -> Optional[Tuple[float, float]]:
        """
        Get joint limits in degrees.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Tuple of (lower_limit, upper_limit) in degrees, or None if not found
        """
        limits_rad = self.joint_limits.get(joint_name)
        if limits_rad is None:
            return None
        
        return (math.degrees(limits_rad[0]), math.degrees(limits_rad[1]))
    
    def get_joint_limits_rad(self, joint_name: str) -> Optional[Tuple[float, float]]:
        """
        Get joint limits in radians.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Tuple of (lower_limit, upper_limit) in radians, or None if not found
        """
        return self.joint_limits.get(joint_name)
    
    def clamp_angle_deg(self, joint_name: str, angle_deg: float) -> float:
        """
        Clamp angle to joint limits in degrees.
        
        Args:
            joint_name: Name of the joint
            angle_deg: Angle in degrees to clamp
            
        Returns:
            Clamped angle in degrees
        """
        limits = self.get_joint_limits_deg(joint_name)
        if limits is None:
            logger.warning(f"No limits found for {joint_name}, returning unclamped angle")
            return angle_deg
        
        lower, upper = limits
        return max(lower, min(upper, angle_deg))
    
    def clamp_angle_rad(self, joint_name: str, angle_rad: float) -> float:
        """
        Clamp angle to joint limits in radians.
        
        Args:
            joint_name: Name of the joint
            angle_rad: Angle in radians to clamp
            
        Returns:
            Clamped angle in radians
        """
        limits = self.get_joint_limits_rad(joint_name)
        if limits is None:
            logger.warning(f"No limits found for {joint_name}, returning unclamped angle")
            return angle_rad
        
        lower, upper = limits
        return max(lower, min(upper, angle_rad))
    
    def validate_angle_deg(self, joint_name: str, angle_deg: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if angle is within joint limits.
        
        Args:
            joint_name: Name of the joint
            angle_deg: Angle in degrees to validate
            
        Returns:
            (is_valid, error_message) tuple
        """
        limits = self.get_joint_limits_deg(joint_name)
        if limits is None:
            return (True, None)  # No limits to validate against
        
        lower, upper = limits
        if angle_deg < lower or angle_deg > upper:
            return (False, f"Angle {angle_deg:.2f}° outside limits [{lower:.2f}°, {upper:.2f}°]")
        
        return (True, None)
    
    def validate_angle_rad(self, joint_name: str, angle_rad: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if angle is within joint limits.
        
        Args:
            joint_name: Name of the joint
            angle_rad: Angle in radians to validate
            
        Returns:
            (is_valid, error_message) tuple
        """
        limits = self.get_joint_limits_rad(joint_name)
        if limits is None:
            return (True, None)  # No limits to validate against
        
        lower, upper = limits
        if angle_rad < lower or angle_rad > upper:
            return (False, f"Angle {math.degrees(angle_rad):.2f}° outside limits [{math.degrees(lower):.2f}°, {math.degrees(upper):.2f}°]")
        
        return (True, None)


def load_urdf(urdf_path: str) -> URDFLoader:
    """
    Load URDF file and return URDFLoader instance.
    
    Args:
        urdf_path: Path to URDF file
        
    Returns:
        URDFLoader instance
    """
    return URDFLoader(urdf_path)

