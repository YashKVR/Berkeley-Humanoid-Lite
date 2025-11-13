"""
Arm Geometry Module

Converts filtered 2D keypoints into arm segment angles (upper arm and forearm)
using vector math and atan2. Handles left/right arms and mirror cases.

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
class ArmAngles:
    """
    Arm segment angles for a single arm.
    
    Attributes:
        upper_arm_angle: Angle of upper arm segment (shoulder to elbow) in degrees
        forearm_angle: Angle of forearm segment (elbow to wrist) in degrees
        shoulder_pitch: Shoulder pitch angle (flexion/extension) in degrees
        shoulder_roll: Shoulder roll angle (abduction/adduction) in degrees
        elbow_angle: Elbow joint angle (bend angle, 180° = straight)
    """
    upper_arm_angle: float  # Upper arm segment angle in image plane (degrees)
    forearm_angle: float     # Forearm segment angle in image plane (degrees)
    shoulder_pitch: float    # Shoulder pitch angle (relative to vertical)
    shoulder_roll: float     # Shoulder roll angle (abduction/adduction)
    elbow_angle: float       # Elbow joint angle (bend angle, 180° = straight)


@dataclass
class ArmKeypoints:
    """
    Arm keypoints for angle calculation.
    
    Attributes:
        shoulder: (x, y) coordinates of shoulder joint
        elbow: (x, y) coordinates of elbow joint
        wrist: (x, y) coordinates of wrist joint
        visibility: Minimum visibility score across all points (0.0-1.0)
    """
    shoulder: Tuple[float, float]
    elbow: Tuple[float, float]
    wrist: Tuple[float, float]
    visibility: float


class ArmGeometry:
    """
    Converts 2D keypoints to arm segment angles.
    
    Handles:
    - Vector-based angle calculation using atan2
    - Left/right arm distinction
    - Mirror mode for camera side view
    - Visibility filtering
    """
    
    # MediaPipe landmark IDs for arm keypoints
    LEFT_SHOULDER_ID = 11
    LEFT_ELBOW_ID = 13
    LEFT_WRIST_ID = 15
    RIGHT_SHOULDER_ID = 12
    RIGHT_ELBOW_ID = 14
    RIGHT_WRIST_ID = 16
    
    def __init__(self, mirror_mode: bool = False, min_visibility: float = 0.5):
        """
        Initialize arm geometry calculator.
        
        Args:
            mirror_mode: If True, mirror the angles (for side-view camera)
            min_visibility: Minimum visibility threshold for keypoints (0.0-1.0)
        """
        self.mirror_mode = mirror_mode
        self.min_visibility = min_visibility
        logger.info(f"ArmGeometry initialized - mirror_mode={mirror_mode}, min_visibility={min_visibility}")
    
    def extract_arm_keypoints(self, landmarks: List) -> Dict[str, Optional[ArmKeypoints]]:
        """
        Extract arm keypoints from MediaPipe landmarks.
        
        Args:
            landmarks: List of PoseLandmark objects from pose estimation
            
        Returns:
            Dictionary with 'left' and 'right' ArmKeypoints, or None if not visible
        """
        # Create a lookup dictionary by landmark_id
        landmark_dict = {lm.landmark_id: lm for lm in landmarks if hasattr(lm, 'landmark_id')}
        
        # Helper to get keypoint or None
        def get_keypoint(shoulder_id: int, elbow_id: int, wrist_id: int) -> Optional[ArmKeypoints]:
            shoulder_lm = landmark_dict.get(shoulder_id)
            elbow_lm = landmark_dict.get(elbow_id)
            wrist_lm = landmark_dict.get(wrist_id)
            
            if not all([shoulder_lm, elbow_lm, wrist_lm]):
                return None
            
            # Type guard: we know these are not None after the check above
            assert shoulder_lm is not None and elbow_lm is not None and wrist_lm is not None
            
            # Check visibility
            min_vis = min(shoulder_lm.visibility, elbow_lm.visibility, wrist_lm.visibility)
            if min_vis < self.min_visibility:
                return None
            
            return ArmKeypoints(
                shoulder=(shoulder_lm.x, shoulder_lm.y),
                elbow=(elbow_lm.x, elbow_lm.y),
                wrist=(wrist_lm.x, wrist_lm.y),
                visibility=min_vis
            )
        
        left_keypoints = get_keypoint(
            self.LEFT_SHOULDER_ID,
            self.LEFT_ELBOW_ID,
            self.LEFT_WRIST_ID
        )
        
        right_keypoints = get_keypoint(
            self.RIGHT_SHOULDER_ID,
            self.RIGHT_ELBOW_ID,
            self.RIGHT_WRIST_ID
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
    
    def calculate_arm_angles(self, keypoints: ArmKeypoints, 
                            arm_side: str = 'left') -> Optional[ArmAngles]:
        """
        Calculate all arm segment angles from keypoints.
        
        Args:
            keypoints: ArmKeypoints object with shoulder, elbow, wrist
            arm_side: 'left' or 'right' (affects mirroring and sign conventions)
            
        Returns:
            ArmAngles object or None if calculation fails
        """
        if keypoints is None:
            return None
        
        # Calculate segment angles
        upper_arm_angle = self.calculate_segment_angle(keypoints.shoulder, keypoints.elbow)
        forearm_angle = self.calculate_segment_angle(keypoints.elbow, keypoints.wrist)
        
        # Calculate joint angles
        # Shoulder pitch: angle of upper arm relative to vertical (0° = straight down)
        # Positive = flexion (arm up), Negative = extension (arm down)
        shoulder_pitch = upper_arm_angle - 90.0
        
        # Shoulder roll: horizontal component (abduction/adduction)
        # For 2D, we approximate using the horizontal component of the upper arm
        # This is a simplified approximation - true roll requires 3D
        dx_upper = keypoints.elbow[0] - keypoints.shoulder[0]
        # For left arm: positive = abduction (away from body), negative = adduction
        # For right arm: positive = adduction (toward body), negative = abduction
        if arm_side == 'left':
            shoulder_roll = math.degrees(math.atan2(dx_upper, abs(keypoints.elbow[1] - keypoints.shoulder[1])))
        else:  # right arm
            shoulder_roll = -math.degrees(math.atan2(dx_upper, abs(keypoints.elbow[1] - keypoints.shoulder[1])))
        
        # Elbow angle: angle between upper arm and forearm segments
        # Calculate the angle difference between the two segments
        angle_diff = abs(forearm_angle - upper_arm_angle)
        # Normalize to 0-180 range (180° = straight, smaller = more bent)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        elbow_angle = 180.0 - angle_diff  # 180° = straight, 0° = fully bent
        
        # Apply mirroring if needed (for side-view camera)
        if self.mirror_mode:
            if arm_side == 'left':
                upper_arm_angle = 180.0 - upper_arm_angle
                forearm_angle = 180.0 - forearm_angle
                shoulder_pitch = -shoulder_pitch
                shoulder_roll = -shoulder_roll
            # Right arm mirroring handled similarly
        
        # Normalize angles to -180 to 180 range
        def normalize_angle(angle: float) -> float:
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return angle
        
        return ArmAngles(
            upper_arm_angle=normalize_angle(upper_arm_angle),
            forearm_angle=normalize_angle(forearm_angle),
            shoulder_pitch=normalize_angle(shoulder_pitch),
            shoulder_roll=normalize_angle(shoulder_roll),
            elbow_angle=elbow_angle
        )
    
    def process_landmarks(self, landmarks: List) -> Dict[str, Optional[ArmAngles]]:
        """
        Process MediaPipe landmarks and return arm angles for both arms.
        
        Args:
            landmarks: List of PoseLandmark objects from pose estimation
            
        Returns:
            Dictionary with 'left' and 'right' ArmAngles, or None if not visible
        """
        keypoints = self.extract_arm_keypoints(landmarks)
        
        left_angles = None
        right_angles = None
        
        if keypoints['left']:
            left_angles = self.calculate_arm_angles(keypoints['left'], arm_side='left')
        
        if keypoints['right']:
            right_angles = self.calculate_arm_angles(keypoints['right'], arm_side='right')
        
        return {
            'left': left_angles,
            'right': right_angles
        }


# Test functions for known poses
def test_arm_extended_pose():
    """Test with synthetic extended arm pose (straight arm)"""
    geometry = ArmGeometry(mirror_mode=False)
    
    # Synthetic landmarks for extended arm pose
    # Shoulder at (100, 100), Elbow at (100, 200), Wrist at (100, 300)
    class MockLandmark:
        def __init__(self, landmark_id, x, y, visibility=1.0):
            self.landmark_id = landmark_id
            self.x = x
            self.y = y
            self.visibility = visibility
    
    landmarks = [
        MockLandmark(11, 100, 100, 1.0),  # LEFT_SHOULDER
        MockLandmark(13, 100, 200, 1.0),  # LEFT_ELBOW
        MockLandmark(15, 100, 300, 1.0),  # LEFT_WRIST
        MockLandmark(12, 120, 100, 1.0),  # RIGHT_SHOULDER
        MockLandmark(14, 120, 200, 1.0),  # RIGHT_ELBOW
        MockLandmark(16, 120, 300, 1.0),  # RIGHT_WRIST
    ]
    
    angles = geometry.process_landmarks(landmarks)
    
    print("Extended arm pose test:")
    if angles['left']:
        print(f"  Left arm - Upper arm: {angles['left'].upper_arm_angle:.1f}°, "
              f"Forearm: {angles['left'].forearm_angle:.1f}°, "
              f"Elbow: {angles['left'].elbow_angle:.1f}°, "
              f"Shoulder pitch: {angles['left'].shoulder_pitch:.1f}°")
    if angles['right']:
        print(f"  Right arm - Upper arm: {angles['right'].upper_arm_angle:.1f}°, "
              f"Forearm: {angles['right'].forearm_angle:.1f}°, "
              f"Elbow: {angles['right'].elbow_angle:.1f}°, "
              f"Shoulder pitch: {angles['right'].shoulder_pitch:.1f}°")
    
    # Expected: upper arm and forearm angles should be ~90° (straight down)
    # Elbow angle should be ~180° (straight arm)
    return angles


def test_arm_bent_pose():
    """Test with synthetic bent arm pose"""
    geometry = ArmGeometry(mirror_mode=False)
    
    class MockLandmark:
        def __init__(self, landmark_id, x, y, visibility=1.0):
            self.landmark_id = landmark_id
            self.x = x
            self.y = y
            self.visibility = visibility
    
    # Left arm with bent elbow: shoulder at (100, 100), elbow at (150, 150), wrist at (200, 200)
    landmarks = [
        MockLandmark(11, 100, 100, 1.0),  # LEFT_SHOULDER
        MockLandmark(13, 150, 150, 1.0),  # LEFT_ELBOW (bent)
        MockLandmark(15, 200, 200, 1.0),  # LEFT_WRIST
        MockLandmark(12, 120, 100, 1.0),  # RIGHT_SHOULDER
        MockLandmark(14, 120, 200, 1.0),  # RIGHT_ELBOW (straight)
        MockLandmark(16, 120, 300, 1.0),  # RIGHT_WRIST
    ]
    
    angles = geometry.process_landmarks(landmarks)
    
    print("\nBent arm pose test:")
    if angles['left']:
        print(f"  Left arm - Upper arm: {angles['left'].upper_arm_angle:.1f}°, "
              f"Forearm: {angles['left'].forearm_angle:.1f}°, "
              f"Elbow: {angles['left'].elbow_angle:.1f}°, "
              f"Shoulder pitch: {angles['left'].shoulder_pitch:.1f}°")
    if angles['right']:
        print(f"  Right arm - Upper arm: {angles['right'].upper_arm_angle:.1f}°, "
              f"Forearm: {angles['right'].forearm_angle:.1f}°, "
              f"Elbow: {angles['right'].elbow_angle:.1f}°, "
              f"Shoulder pitch: {angles['right'].shoulder_pitch:.1f}°")
    
    # Expected: left elbow angle should be < 180° (bent)
    # Right arm should be straight (~180°)
    return angles


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Arm Geometry Module Tests")
    print("=" * 60)
    
    test_arm_extended_pose()
    test_arm_bent_pose()
    
    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)