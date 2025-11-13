"""
MediaPipe Pose Estimation Module

This module provides human pose estimation using MediaPipe for skeleton joint
detection, posture analysis, and gesture recognition.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PoseLandmark:
    """
    Pose landmark container for individual joint information.
    
    Attributes:
        landmark_id: MediaPipe landmark ID (0-32)
        name: Human-readable landmark name
        x: X coordinate in pixels
        y: Y coordinate in pixels
        z: Z coordinate (relative depth)
        visibility: Landmark visibility score (0.0-1.0)
    """
    landmark_id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    """
    MediaPipe-based pose estimator for human skeleton joint detection.
    
    Provides real-time pose estimation with 33 landmarks for posture
    and gesture recognition applications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MediaPipe pose estimator.
        
        Args:
            config: Configuration dictionary for pose estimation parameters
        """
        self.config = config or {}
        
        # MediaPipe configuration
        self.static_image_mode = self.config.get('static_image_mode', False)
        self.model_complexity = self.config.get('model_complexity', 1)
        self.smooth_landmarks = self.config.get('smooth_landmarks', True)
        self.enable_segmentation = self.config.get('enable_segmentation', False)
        self.smooth_segmentation = self.config.get('smooth_segmentation', True)
        self.min_detection_confidence = self.config.get('min_detection_confidence', 0.5)
        self.min_tracking_confidence = self.config.get('min_tracking_confidence', 0.5)
        
        # Initialize MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Landmark names mapping
        self.landmark_names = {
            0: "NOSE",
            1: "LEFT_EYE_INNER", 2: "LEFT_EYE", 3: "LEFT_EYE_OUTER",
            4: "RIGHT_EYE_INNER", 5: "RIGHT_EYE", 6: "RIGHT_EYE_OUTER",
            7: "LEFT_EAR", 8: "RIGHT_EAR",
            9: "MOUTH_LEFT", 10: "MOUTH_RIGHT",
            11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER",
            13: "LEFT_ELBOW", 14: "RIGHT_ELBOW",
            15: "LEFT_WRIST", 16: "RIGHT_WRIST",
            17: "LEFT_PINKY", 18: "RIGHT_PINKY",
            19: "LEFT_INDEX", 20: "RIGHT_INDEX",
            21: "LEFT_THUMB", 22: "RIGHT_THUMB",
            23: "LEFT_HIP", 24: "RIGHT_HIP",
            25: "LEFT_KNEE", 26: "RIGHT_KNEE",
            27: "LEFT_ANKLE", 28: "RIGHT_ANKLE",
            29: "LEFT_HEEL", 30: "RIGHT_HEEL",
            31: "LEFT_FOOT_INDEX", 32: "RIGHT_FOOT_INDEX"
        }
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        
        logger.info(f"PoseEstimator initialized - Model complexity: {self.model_complexity}, "
                   f"Detection confidence: {self.min_detection_confidence}")
    
    def _convert_to_pixel_coordinates(self, landmark, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Convert normalized landmark coordinates to pixel coordinates.
        
        Args:
            landmark: MediaPipe landmark object
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        x = landmark.x * image_width
        y = landmark.y * image_height
        return x, y
    
    def estimate_pose(self, frame: np.ndarray) -> List[PoseLandmark]:
        """
        Estimate human pose from the given frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of PoseLandmark objects (33 landmarks when human detected)
        """
        try:
            # Convert BGR to RGB (MediaPipe requirement)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            image_height, image_width = frame.shape[:2]
            
            # Process frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            landmarks = []
            
            if results.pose_landmarks:
                # Convert landmarks to pixel coordinates
                for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                    x, y = self._convert_to_pixel_coordinates(landmark, image_width, image_height)
                    
                    pose_landmark = PoseLandmark(
                        landmark_id=landmark_id,
                        name=self.landmark_names.get(landmark_id, f"LANDMARK_{landmark_id}"),
                        x=x,
                        y=y,
                        z=landmark.z,
                        visibility=landmark.visibility
                    )
                    
                    landmarks.append(pose_landmark)
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error during pose estimation: {e}")
            return []
    
    def get_pose_info(self, landmarks: List[PoseLandmark]) -> Dict[str, Any]:
        """
        Get pose information and statistics.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Dictionary with pose information
        """
        if not landmarks:
            return {
                'landmarks_count': 0,
                'pose_detected': False,
                'avg_visibility': 0.0
            }
        
        avg_visibility = sum(landmark.visibility for landmark in landmarks) / len(landmarks)
        
        return {
            'landmarks_count': len(landmarks),
            'pose_detected': True,
            'avg_visibility': avg_visibility,
            'landmarks': landmarks
        }
    
    def draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[PoseLandmark]) -> np.ndarray:
        """
        Draw pose landmarks on the frame.
        
        Args:
            frame: Input frame
            landmarks: List of pose landmarks
            
        Returns:
            Frame with drawn landmarks
        """
        result_frame = frame.copy()
        
        if landmarks:
            # Draw landmarks as simple circles
            for landmark in landmarks:
                if landmark.visibility > 0.5:  # Only draw visible landmarks
                    cv2.circle(result_frame, (int(landmark.x), int(landmark.y)), 3, (0, 255, 0), -1)
                    cv2.putText(result_frame, landmark.name, (int(landmark.x) + 5, int(landmark.y) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return result_frame
    
    def get_key_joints(self, landmarks: List[PoseLandmark]) -> Dict[str, Optional[PoseLandmark]]:
        """
        Get key joints for posture analysis.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Dictionary with key joint landmarks
        """
        key_joints = {}
        
        if landmarks:
            # Map landmark IDs to key joint names
            key_joint_mapping = {
                'nose': 0,
                'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14,
                'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24,
                'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28
            }
            
            for joint_name, landmark_id in key_joint_mapping.items():
                if landmark_id < len(landmarks):
                    key_joints[joint_name] = landmarks[landmark_id]
                else:
                    key_joints[joint_name] = None
        
        return key_joints
    
    def calculate_pose_angles(self, landmarks: List[PoseLandmark]) -> Dict[str, float]:
        """
        Calculate key pose angles for posture analysis.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Dictionary with calculated angles
        """
        angles = {}
        
        if len(landmarks) >= 33:
            # Calculate elbow angles
            left_shoulder = landmarks[11]
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]
            
            right_shoulder = landmarks[12]
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]
            
            # Calculate angles using dot product
            def calculate_angle(p1, p2, p3):
                # Vector from p2 to p1
                v1 = np.array([p1.x - p2.x, p1.y - p2.y])
                # Vector from p2 to p3
                v2 = np.array([p3.x - p2.x, p3.y - p2.y])
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                return np.degrees(angle)
            
            # Calculate elbow angles
            if (left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and 
                left_wrist.visibility > 0.5):
                angles['left_elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if (right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5 and 
                right_wrist.visibility > 0.5):
                angles['right_elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        return angles
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': 'MediaPipe Pose',
            'model_complexity': self.model_complexity,
            'static_image_mode': self.static_image_mode,
            'smooth_landmarks': self.smooth_landmarks,
            'enable_segmentation': self.enable_segmentation,
            'min_detection_confidence': self.min_detection_confidence,
            'min_tracking_confidence': self.min_tracking_confidence,
            'landmarks_count': 33
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update pose estimation configuration.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Update parameters
        self.static_image_mode = self.config.get('static_image_mode', False)
        self.model_complexity = self.config.get('model_complexity', 1)
        self.smooth_landmarks = self.config.get('smooth_landmarks', True)
        self.enable_segmentation = self.config.get('enable_segmentation', False)
        self.smooth_segmentation = self.config.get('smooth_segmentation', True)
        self.min_detection_confidence = self.config.get('min_detection_confidence', 0.5)
        self.min_tracking_confidence = self.config.get('min_tracking_confidence', 0.5)
        
        # Reinitialize pose solution with new parameters
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        logger.info(f"Configuration updated: {new_config}")
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


# Example usage and testing
if __name__ == "__main__":
    from camera_stream import CameraStream
    
    # Configuration
    config = {
        'model_complexity': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'smooth_landmarks': True
    }
    
    # Create pose estimator
    pose_estimator = PoseEstimator(config)
    
    # Create camera stream
    camera_config = {
        'width': 640,
        'height': 480,
        'fps': 30
    }
    
    camera = CameraStream(camera_id=0, config=camera_config)
    
    try:
        # Start camera
        if camera.start():
            print("Camera and pose estimator started successfully")
            print("Press 'q' to quit, 's' for stats")
            
            frame_count = 0
            pose_count = 0
            start_time = time.time()
            
            while True:
                frame = camera.read()
                
                if frame is not None:
                    frame_count += 1
                    
                    # Estimate pose
                    landmarks = pose_estimator.estimate_pose(frame)
                    
                    if landmarks:
                        pose_count += 1
                        
                        # Get pose information
                        pose_info = pose_estimator.get_pose_info(landmarks)
                        
                        # Get key joints
                        key_joints = pose_estimator.get_key_joints(landmarks)
                        
                        # Calculate angles
                        angles = pose_estimator.calculate_pose_angles(landmarks)
                        
                        # Count visible landmarks
                        visible_landmarks = sum(1 for landmark in landmarks if landmark.visibility > 0.3)
                        
                        # Draw landmarks
                        result_frame = pose_estimator.draw_pose_landmarks(frame, landmarks)
                        
                        # Add landmark count overlay
                        cv2.putText(result_frame, f"Landmarks: {visible_landmarks}/33", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Total landmarks: {len(landmarks)}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Display frame
                        cv2.imshow('MediaPipe Pose Estimation', result_frame)
                        
                        # Optional: Print pose information
                        if frame_count % 30 == 0:  # Print every 30 frames
                            visible_count = sum(1 for landmark in landmarks if landmark.visibility > 0.3)
                            print(f"Pose detected: {visible_count}/33 landmarks, "
                                  f"Total: {len(landmarks)}, Avg visibility: {pose_info['avg_visibility']:.3f}")
                            if angles:
                                print(f"Angles: {angles}")
                    else:
                        # No pose detected
                        cv2.putText(frame, "No Pose Detected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "Landmarks: 0/33", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.imshow('MediaPipe Pose Estimation', frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    elif key == ord('s'):
                        # Show statistics
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        print(f"\n=== Pose Estimation Statistics ===")
                        print(f"Frames processed: {frame_count}")
                        print(f"Poses detected: {pose_count}")
                        print(f"Elapsed time: {elapsed:.2f}s")
                        print(f"Processing FPS: {fps:.2f}")
                        print(f"Pose detection rate: {pose_count/frame_count:.2f}")
                        print(f"Model info: {pose_estimator.get_model_info()}")
                        print("===================================\n")
                
                else:
                    time.sleep(0.01)
        
        else:
            print("Failed to start camera")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n=== Final Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total poses detected: {pose_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Pose detection rate: {pose_count/frame_count:.2f}")
        print("=======================")