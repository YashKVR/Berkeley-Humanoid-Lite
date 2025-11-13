"""
Leg Geometry Module

Converts filtered 2D keypoints into leg segment angles (thigh and shank)
using vector math and atan2. Handles left/right legs and mirror cases.

This module provides a clean separation between vision (pixels) and geometry (angles),
enabling unit-agnostic signals that can be scaled and mapped per joint.
"""

import numpy as np
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LegAngles:
    """
    Leg segment angles for a single leg.
    
    Attributes:
        thigh_angle: Angle of thigh segment (hip to knee) in degrees
        shank_angle: Angle of shank segment (knee to ankle) in degrees
        hip_angle: Angle at hip joint (relative to vertical) in degrees
        knee_angle: Angle at knee joint (bend angle) in degrees
    """
    thigh_angle: float  # Thigh segment angle in image plane (degrees)
    shank_angle: float   # Shank segment angle in image plane (degrees)
    hip_angle: float     # Hip joint angle (relative to vertical)
    knee_angle: float    # Knee joint angle (bend angle, 180° = straight)


@dataclass
class LegKeypoints:
    """
    Leg keypoints for angle calculation.
    
    Attributes:
        hip: (x, y) coordinates of hip joint
        knee: (x, y) coordinates of knee joint
        ankle: (x, y) coordinates of ankle joint
        visibility: Minimum visibility score across all points (0.0-1.0)
    """
    hip: Tuple[float, float]
    knee: Tuple[float, float]
    ankle: Tuple[float, float]
    visibility: float


class LegGeometry:
    """
    Converts 2D keypoints to leg segment angles.
    
    Handles:
    - Vector-based angle calculation using atan2
    - Left/right leg distinction
    - Mirror mode for camera side view
    - Visibility filtering
    """
    
    # MediaPipe landmark IDs for leg keypoints
    LEFT_HIP_ID = 23
    LEFT_KNEE_ID = 25
    LEFT_ANKLE_ID = 27
    RIGHT_HIP_ID = 24
    RIGHT_KNEE_ID = 26
    RIGHT_ANKLE_ID = 28
    
    def __init__(self, mirror_mode: bool = False, min_visibility: float = 0.5):
        """
        Initialize leg geometry calculator.
        
        Args:
            mirror_mode: If True, mirror the angles (for side-view camera)
            min_visibility: Minimum visibility threshold for keypoints (0.0-1.0)
        """
        self.mirror_mode = mirror_mode
        self.min_visibility = min_visibility
        logger.info(f"LegGeometry initialized - mirror_mode={mirror_mode}, min_visibility={min_visibility}")
    
    def extract_leg_keypoints(self, landmarks: List) -> Dict[str, Optional[LegKeypoints]]:
        """
        Extract leg keypoints from MediaPipe landmarks.
        
        Args:
            landmarks: List of PoseLandmark objects from pose estimation
            
        Returns:
            Dictionary with 'left' and 'right' LegKeypoints, or None if not visible
        """
        # Create a lookup dictionary by landmark_id
        landmark_dict = {lm.landmark_id: lm for lm in landmarks if hasattr(lm, 'landmark_id')}
        
        # Helper to get keypoint or None
        def get_keypoint(hip_id: int, knee_id: int, ankle_id: int) -> Optional[LegKeypoints]:
            hip_lm = landmark_dict.get(hip_id)
            knee_lm = landmark_dict.get(knee_id)
            ankle_lm = landmark_dict.get(ankle_id)
            
            if not all([hip_lm, knee_lm, ankle_lm]):
                return None
            
            # Check visibility
            min_vis = min(hip_lm.visibility, knee_lm.visibility, ankle_lm.visibility)
            if min_vis < self.min_visibility:
                return None
            
            return LegKeypoints(
                hip=(hip_lm.x, hip_lm.y),
                knee=(knee_lm.x, knee_lm.y),
                ankle=(ankle_lm.x, ankle_lm.y),
                visibility=min_vis
            )
        
        left_keypoints = get_keypoint(
            self.LEFT_HIP_ID,
            self.LEFT_KNEE_ID,
            self.LEFT_ANKLE_ID
        )
        
        right_keypoints = get_keypoint(
            self.RIGHT_HIP_ID,
            self.RIGHT_KNEE_ID,
            self.RIGHT_ANKLE_ID
        )
        
        return {
            'left': left_keypoints,
            'right': right_keypoints
        }
    
    def calculate_segment_angle(self, point1: Tuple[float, float], 
                                point2: Tuple[float, float]) -> float:
        """
        Calculate angle of a segment (vector) in the image plane.
        
        Args:
            point1: Starting point (x, y)
            point2: Ending point (x, y)
            
        Returns:
            Angle in degrees, measured from horizontal (0° = right, 90° = down)
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # Use atan2 to get angle in radians, then convert to degrees
        # atan2(y, x) gives angle from positive x-axis
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_leg_angles(self, keypoints: LegKeypoints, 
                            leg_side: str = 'left') -> Optional[LegAngles]:
        """
        Calculate all leg segment angles from keypoints.
        
        Args:
            keypoints: LegKeypoints object with hip, knee, ankle
            leg_side: 'left' or 'right' (affects mirroring)
            
        Returns:
            LegAngles object or None if calculation fails
        """
        if keypoints is None:
            return None
        
        # Calculate segment angles
        thigh_angle = self.calculate_segment_angle(keypoints.hip, keypoints.knee)
        shank_angle = self.calculate_segment_angle(keypoints.knee, keypoints.ankle)
        
        # Calculate joint angles
        # Hip angle: angle of thigh relative to vertical (0° = straight down)
        hip_angle = thigh_angle - 90.0
        
        # Knee angle: angle between thigh and shank segments
        # Calculate the angle difference between the two segments
        angle_diff = abs(shank_angle - thigh_angle)
        # Normalize to 0-180 range (180° = straight, smaller = more bent)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        knee_angle = 180.0 - angle_diff  # 180° = straight, 0° = fully bent
        
        # Apply mirroring if needed (for side-view camera)
        if self.mirror_mode:
            if leg_side == 'left':
                thigh_angle = 180.0 - thigh_angle
                shank_angle = 180.0 - shank_angle
                hip_angle = -hip_angle
            # Right leg mirroring handled similarly
        
        # Normalize angles to -180 to 180 range
        def normalize_angle(angle: float) -> float:
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return angle
        
        return LegAngles(
            thigh_angle=normalize_angle(thigh_angle),
            shank_angle=normalize_angle(shank_angle),
            hip_angle=normalize_angle(hip_angle),
            knee_angle=knee_angle
        )
    
    def process_landmarks(self, landmarks: List) -> Dict[str, Optional[LegAngles]]:
        """
        Process MediaPipe landmarks and return leg angles for both legs.
        
        Args:
            landmarks: List of PoseLandmark objects from pose estimation
            
        Returns:
            Dictionary with 'left' and 'right' LegAngles, or None if not visible
        """
        keypoints = self.extract_leg_keypoints(landmarks)
        
        left_angles = None
        right_angles = None
        
        if keypoints['left']:
            left_angles = self.calculate_leg_angles(keypoints['left'], leg_side='left')
        
        if keypoints['right']:
            right_angles = self.calculate_leg_angles(keypoints['right'], leg_side='right')
        
        return {
            'left': left_angles,
            'right': right_angles
        }


# Test functions for known poses
def test_stand_pose():
    """Test with synthetic stand pose (straight legs)"""
    geometry = LegGeometry(mirror_mode=False)
    
    # Synthetic landmarks for standing pose
    # Hip at (100, 100), Knee at (100, 200), Ankle at (100, 300)
    class MockLandmark:
        def __init__(self, landmark_id, x, y, visibility=1.0):
            self.landmark_id = landmark_id
            self.x = x
            self.y = y
            self.visibility = visibility
    
    landmarks = [
        MockLandmark(23, 100, 100, 1.0),  # LEFT_HIP
        MockLandmark(25, 100, 200, 1.0),  # LEFT_KNEE
        MockLandmark(27, 100, 300, 1.0),  # LEFT_ANKLE
        MockLandmark(24, 120, 100, 1.0),  # RIGHT_HIP
        MockLandmark(26, 120, 200, 1.0),  # RIGHT_KNEE
        MockLandmark(28, 120, 300, 1.0),  # RIGHT_ANKLE
    ]
    
    angles = geometry.process_landmarks(landmarks)
    
    print("Stand pose test:")
    if angles['left']:
        print(f"  Left leg - Thigh: {angles['left'].thigh_angle:.1f}°, "
              f"Shank: {angles['left'].shank_angle:.1f}°, "
              f"Knee: {angles['left'].knee_angle:.1f}°")
    if angles['right']:
        print(f"  Right leg - Thigh: {angles['right'].thigh_angle:.1f}°, "
              f"Shank: {angles['right'].shank_angle:.1f}°, "
              f"Knee: {angles['right'].knee_angle:.1f}°")
    
    # Expected: thigh and shank angles should be ~90° (straight down)
    # Knee angle should be ~180° (straight leg)
    return angles


def test_knee_lift_pose():
    """Test with synthetic knee-lift pose"""
    geometry = LegGeometry(mirror_mode=False)
    
    class MockLandmark:
        def __init__(self, landmark_id, x, y, visibility=1.0):
            self.landmark_id = landmark_id
            self.x = x
            self.y = y
            self.visibility = visibility
    
    # Left leg with bent knee: hip at (100, 100), knee at (150, 150), ankle at (200, 200)
    landmarks = [
        MockLandmark(23, 100, 100, 1.0),  # LEFT_HIP
        MockLandmark(25, 150, 150, 1.0),  # LEFT_KNEE (bent)
        MockLandmark(27, 200, 200, 1.0),  # LEFT_ANKLE
        MockLandmark(24, 120, 100, 1.0),  # RIGHT_HIP
        MockLandmark(26, 120, 200, 1.0),  # RIGHT_KNEE (straight)
        MockLandmark(28, 120, 300, 1.0),  # RIGHT_ANKLE
    ]
    
    angles = geometry.process_landmarks(landmarks)
    
    print("\nKnee-lift pose test:")
    if angles['left']:
        print(f"  Left leg - Thigh: {angles['left'].thigh_angle:.1f}°, "
              f"Shank: {angles['left'].shank_angle:.1f}°, "
              f"Knee: {angles['left'].knee_angle:.1f}°")
    if angles['right']:
        print(f"  Right leg - Thigh: {angles['right'].thigh_angle:.1f}°, "
              f"Shank: {angles['right'].shank_angle:.1f}°, "
              f"Knee: {angles['right'].knee_angle:.1f}°")
    
    # Expected: left knee angle should be < 180° (bent)
    # Right leg should be straight (~180°)
    return angles


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Leg Geometry Module Tests")
    print("=" * 60)
    
    test_stand_pose()
    test_knee_lift_pose()
    
    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)

