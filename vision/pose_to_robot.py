#!/usr/bin/env python
#
# Pose to Robot Retargeting Script
# 
# Real-time human pose estimation → arm geometry → robot joint retargeting → motor control
# Captures human arm movement and retargets it to the robot's right arm in real-time
#

import sys
import os
import argparse
import time
import cv2
import logging
import math
import numpy as np
from typing import Optional, Dict

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add project root to path for package imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import vision modules
from vision.pose2d_mediapipe import PoseEstimator, PoseLandmark  # type: ignore
from vision.arm_geometry import ArmGeometry  # type: ignore
from vision.camera_stream import CameraStream  # type: ignore
from vision.retarget import RetargetMapper  # type: ignore

# Import motor control
from loop_rate_limiters import RateLimiter
from vision.arm_motor_controller import ArmMotorController  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseToRobot:
    """
    Main class for pose-to-robot retargeting pipeline for right arm.
    
    Handles:
    - Camera capture
    - Pose estimation
    - Arm geometry calculation
    - Robot joint retargeting
    - Motor control via CAN bus
    """
    
    # Right arm joint to motor device ID mapping
    # Note: All motors communicate on the same CAN bus channel (can0)
    # These are the device IDs (motor IDs) on that bus: 2, 4, 6, 8, 10
    JOINT_TO_CAN_ID = {
        'arm_right_shoulder_pitch_joint': 2,
        'arm_right_shoulder_roll_joint': 4,
        'arm_right_shoulder_yaw_joint': 6,
        'arm_right_elbow_pitch_joint': 8,
        'arm_right_elbow_roll_joint': 10,
    }
    
    def __init__(
        self,
        camera_id: int = 0,
        can_channel: str = 'can0',
        urdf_path: Optional[str] = None,
        retarget_config_path: Optional[str] = None,
        mirror_mode: bool = False,
        min_visibility: float = 0.5,
        update_rate: float = 100.0,  # Hz (higher rate for smoother motion, like move_actuator uses 200 Hz)
        enable_motor_control: bool = True,
        min_movement_threshold: float = 0.01  # Minimum movement in radians to send command
    ):
        """
        Initialize pose-to-robot retargeting system.
        
        Args:
            camera_id: Camera device ID
            can_channel: CAN bus channel (e.g., 'can0')
            urdf_path: Path to URDF file
            retarget_config_path: Path to retarget config YAML
            mirror_mode: Mirror angles for side-view camera
            min_visibility: Minimum visibility threshold for keypoints
            update_rate: Motor update rate in Hz
            enable_motor_control: If False, only simulate (don't send to motors)
            min_movement_threshold: Minimum movement in radians to send command
        """
        self.camera_id = camera_id
        self.can_channel = can_channel
        self.mirror_mode = mirror_mode
        self.min_visibility = min_visibility
        self.update_rate = update_rate
        self.enable_motor_control = enable_motor_control
        self.min_movement_threshold = min_movement_threshold
        
        # Default paths
        if urdf_path is None:
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
        if retarget_config_path is None:
            retarget_config_path = os.path.join(project_root, 'configs', 'retarget.yaml')
        
        self.urdf_path = urdf_path
        self.retarget_config_path = retarget_config_path
        
        # Initialize components
        logger.info("Initializing pose-to-robot system...")
        
        # Initialize pose estimator
        pose_config = {
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'smooth_landmarks': True
        }
        self.pose_estimator = PoseEstimator(config=pose_config)
        
        # Initialize arm geometry
        self.arm_geometry = ArmGeometry(mirror_mode=mirror_mode, min_visibility=min_visibility)
        
        # Initialize retarget mapper
        self.retarget_mapper = RetargetMapper(urdf_path, retarget_config_path)
        
        # Initialize camera
        camera_config = {
            'width': 640,
            'height': 480,
            'fps': 30
        }
        self.camera = CameraStream(camera_id=camera_id, config=camera_config)
        
        # Initialize motor controller
        self.motor_controller = None
        if enable_motor_control:
            logger.info(f"Initializing motor controller with CAN channel: {can_channel}")
            try:
                self.motor_controller = ArmMotorController(can_channel=can_channel)
                logger.info("ArmMotorController created, initializing motors...")
                if not self.motor_controller.initialize_motors():
                    logger.error("Failed to initialize motors")
                    logger.warning("Continuing in simulation mode (no motor control)")
                    self.enable_motor_control = False
                    self.motor_controller = None
                else:
                    logger.info("Motor controller initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize motor controller: {e}", exc_info=True)
                logger.warning("Continuing in simulation mode (no motor control)")
                self.enable_motor_control = False
                self.motor_controller = None
        else:
            logger.info("Motor control disabled at initialization")
        
        # Rate limiter for camera/processing
        self.rate_limiter = RateLimiter(frequency=60.0)  # Camera/processing rate
        
        # Statistics
        self.frame_count = 0
        self.pose_detected_count = 0
        self.start_time = None
        
        logger.info("Pose-to-robot system initialized")
    
    def process_frame(self, frame) -> Optional[Dict[str, Optional[float]]]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Dictionary of joint commands (joint_name -> angle_rad) or None
        """
        # Estimate pose
        landmarks = self.pose_estimator.estimate_pose(frame)
        
        if not landmarks:
            return None
        
        # Calculate arm angles (right arm only)
        arm_angles_dict = self.arm_geometry.process_landmarks(landmarks)
        right_arm_angles = arm_angles_dict.get('right')
        
        if not right_arm_angles:
            return None
        
        # Retarget to robot joints (returns angles in degrees)
        joint_commands_deg = self.retarget_mapper.retarget_arm_angles(right_arm_angles)
        
        # Convert to radians
        joint_commands_rad = {}
        for joint_name, angle_deg in joint_commands_deg.items():
            if angle_deg is not None:
                joint_commands_rad[joint_name] = math.radians(angle_deg)
            else:
                joint_commands_rad[joint_name] = None
        
        return joint_commands_rad
    
    def run(self, display: bool = True):
        """
        Run the main retargeting loop.
        
        Args:
            display: If True, display video feed with overlays
        """
        logger.info("Starting pose-to-robot retargeting...")
        logger.info("Press 'q' to quit, 's' for stats, 'r' to reset stats")
        
        # Start camera FIRST (before motors, so we can see what's happening)
        logger.info("Initializing camera...")
        if not self.camera.start():
            logger.error("Failed to start camera")
            logger.error("Please check:")
            logger.error("  1. Camera device is connected and available")
            logger.error("  2. Camera is not being used by another application")
            logger.error("  3. Camera permissions are correct")
            # Don't start motors if camera failed
            return
        
        logger.info("Camera started successfully")
        
        # Start motor controller if enabled (after camera is working)
        if self.enable_motor_control and self.motor_controller:
            logger.info("Starting motor control loop...")
            try:
                self.motor_controller.start()
                logger.info("Motor control loop started")
            except Exception as e:
                logger.error(f"Failed to start motor control loop: {e}", exc_info=True)
                logger.warning("Continuing without motor control")
                self.enable_motor_control = False
        else:
            logger.info("Motor control disabled - running in simulation mode")
        
        self.start_time = time.time()
        
        try:
            while True:
                # Read frame
                frame = self.camera.read()
                if frame is None:
                    continue
                
                self.frame_count += 1
                
                # Process frame
                landmarks = self.pose_estimator.estimate_pose(frame)
                joint_commands = None
                
                if landmarks:
                    self.pose_detected_count += 1
                    # Calculate arm angles
                    arm_angles_dict = self.arm_geometry.process_landmarks(landmarks)
                    right_arm_angles = arm_angles_dict.get('right')
                    
                    # Debug: Print arm angles
                    if self.frame_count % 30 == 0 and right_arm_angles:  # Print every 30 frames
                        logger.info(f"Right arm - Shoulder pitch: {right_arm_angles.shoulder_pitch:.1f}°, "
                                  f"Shoulder roll: {right_arm_angles.shoulder_roll:.1f}°, "
                                  f"Elbow: {right_arm_angles.elbow_angle:.1f}°, "
                                  f"Upper arm: {right_arm_angles.upper_arm_angle:.1f}°, "
                                  f"Forearm: {right_arm_angles.forearm_angle:.1f}°")
                    
                    # Retarget to robot joints
                    joint_commands_deg = self.retarget_mapper.retarget_arm_angles(right_arm_angles)
                    
                    # Convert to radians
                    joint_commands = {}
                    for joint_name, angle_deg in joint_commands_deg.items():
                        if angle_deg is not None:
                            joint_commands[joint_name] = math.radians(angle_deg)
                        else:
                            joint_commands[joint_name] = None
                    
                    # Debug: Print joint commands
                    if self.frame_count % 30 == 0 and joint_commands:
                        logger.info("Joint commands (rad):")
                        for joint_name, angle in joint_commands.items():
                            if angle is not None:
                                logger.info(f"  {joint_name}: {angle:.3f} rad ({math.degrees(angle):.2f}°)")
                
                # Update motor target angles (non-blocking, controller handles interpolation)
                if joint_commands and self.enable_motor_control and self.motor_controller:
                    self.motor_controller.update_target_angles(joint_commands)
                elif joint_commands and not self.enable_motor_control:
                    # Debug: Log when we have commands but motor control is disabled
                    if self.frame_count % 100 == 0:  # Log every 100 frames
                        logger.debug(f"Joint commands available but motor control disabled (enable_motor_control={self.enable_motor_control}, motor_controller={self.motor_controller is not None})")
                elif joint_commands and self.enable_motor_control and not self.motor_controller:
                    # Debug: Log when motor control is enabled but controller is None
                    if self.frame_count % 100 == 0:
                        logger.warning(f"Motor control enabled but motor_controller is None!")
                
                # Rate limiting (for camera/processing rate, not motor rate)
                self.rate_limiter.sleep()
                
                # Display frame
                if display:
                    display_frame = frame.copy()
                    
                    # Draw pose landmarks if available
                    if landmarks:
                        display_frame = self.pose_estimator.draw_pose_landmarks(display_frame, landmarks)
                    
                    # Draw stats
                    current_time = time.time()
                    fps = self.frame_count / (current_time - self.start_time) if self.start_time else 0
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Poses: {self.pose_detected_count}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if joint_commands:
                        # Show joint commands
                        y_offset = 120
                        for i, (joint_name, angle) in enumerate(list(joint_commands.items())[:5]):
                            if angle is not None:
                                text = f"{joint_name}: {math.degrees(angle):.1f}°"
                                cv2.putText(display_frame, text, (10, y_offset + i * 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    cv2.imshow('Pose to Robot - Right Arm', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested")
                        break
                    elif key == ord('s'):
                        self._print_stats()
                    elif key == ord('r'):
                        self._reset_stats()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Cleanup
            logger.info("Stopping camera and motor controller...")
            self.camera.stop()
            if self.enable_motor_control and self.motor_controller:
                self.motor_controller.stop()
            cv2.destroyAllWindows()
            
            self._print_stats()
            logger.info("Pose-to-robot system stopped")
    
    def _print_stats(self):
        """Print statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print("Statistics")
        print("=" * 60)
        print(f"Frames processed: {self.frame_count}")
        print(f"Poses detected: {self.pose_detected_count}")
        print(f"Average FPS: {fps:.2f}")
        if self.frame_count > 0:
            print(f"Pose detection rate: {100 * self.pose_detected_count / self.frame_count:.1f}%")
        print("=" * 60)
    
    def _reset_stats(self):
        """Reset statistics"""
        self.frame_count = 0
        self.pose_detected_count = 0
        self.start_time = time.time()
        logger.info("Statistics reset")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time human arm pose retargeting to robot right arm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  python vision/pose_to_robot.py
  
  # Custom camera and CAN channel
  python vision/pose_to_robot.py --camera 1 --can can1
  
  # Simulation mode (no motor control)
  python vision/pose_to_robot.py --no-motor
  
  # Lower update rate for stability
  python vision/pose_to_robot.py --update-rate 20
        '''
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--can', type=str, default='can0',
                       help='CAN bus channel (default: can0)')
    parser.add_argument('--urdf', type=str, default=None,
                       help='Path to URDF file')
    parser.add_argument('--retarget-config', type=str, default=None,
                       help='Path to retarget config YAML')
    parser.add_argument('--mirror', action='store_true',
                       help='Enable mirror mode for side-view camera')
    parser.add_argument('--min-visibility', type=float, default=0.5,
                       help='Minimum visibility threshold for keypoints (default: 0.5)')
    parser.add_argument('--update-rate', type=float, default=30.0,
                       help='Motor update rate in Hz (default: 30.0)')
    parser.add_argument('--no-motor', action='store_true',
                       help='Run in simulation mode (no motor control)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        # Enable debug logging for retarget module
        logging.getLogger('vision.retarget').setLevel(logging.DEBUG)
    
    # Create and run pose-to-robot system
    system = PoseToRobot(
        camera_id=args.camera,
        can_channel=args.can,
        urdf_path=args.urdf,
        retarget_config_path=args.retarget_config,
        mirror_mode=args.mirror,
        min_visibility=args.min_visibility,
        update_rate=args.update_rate,
        enable_motor_control=not args.no_motor
    )
    
    system.run(display=not args.no_display)


if __name__ == "__main__":
    main()
