"""
Threaded Camera Stream Module

This module provides a thread-safe camera capture system using OpenCV
for stable, real-time frame streaming with configurable parameters.
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe camera stream using OpenCV VideoCapture.
    
    Provides continuous frame streaming with background thread processing
    and thread-safe access to frames.
    """
    
    def __init__(self, camera_id: int = 0, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the camera stream.
        
        Args:
            camera_id: Camera device ID (default: 0)
            config: Configuration dictionary with camera parameters
        """
        self.camera_id = camera_id
        self.config = config or {}
        
        # Default configuration
        self.width = self.config.get('width', 1280)
        self.height = self.config.get('height', 720)
        self.fps = self.config.get('fps', 30)
        
        # Camera and threading components
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = None
        
        logger.info(f"CameraStream initialized with config: {self.config}")
    
    def _initialize_camera(self) -> bool:
        """
        Initialize the OpenCV VideoCapture with configured parameters.
        
        Returns:
            bool: True if camera initialization successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized - Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def _capture_frames(self):
        """
        Background thread function to continuously capture frames.
        """
        logger.info("Camera capture thread started")
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.error("Camera not available")
                    break
                
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Thread-safe frame update
                with self.lock:
                    self.frame = frame.copy()
                    self.frame_count += 1
                    self.last_frame_time = time.time()
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Error in capture thread: {e}")
                break
        
        logger.info("Camera capture thread stopped")
    
    def start(self) -> bool:
        """
        Start the camera stream.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Camera stream is already running")
            return True
        
        # Initialize camera
        if not self._initialize_camera():
            return False
        
        # Start capture thread
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        
        # Wait for first frame
        timeout = 5.0  # 5 second timeout
        start_wait = time.time()
        
        while self.frame is None and (time.time() - start_wait) < timeout:
            time.sleep(0.1)
        
        if self.frame is None:
            logger.error("Failed to capture first frame within timeout")
            self.stop()
            return False
        
        logger.info("Camera stream started successfully")
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read the latest frame from the camera stream.
        
        Returns:
            np.ndarray: Latest frame or None if no frame available
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def stop(self):
        """
        Stop the camera stream and release resources.
        """
        if not self.running:
            logger.warning("Camera stream is not running")
            return
        
        logger.info("Stopping camera stream...")
        
        # Stop capture thread
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame
        with self.lock:
            self.frame = None
        
        # Log performance statistics
        if self.start_time:
            duration = time.time() - self.start_time
            avg_fps = self.frame_count / duration if duration > 0 else 0
            logger.info(f"Camera stream stopped - Duration: {duration:.2f}s, "
                       f"Frames: {self.frame_count}, Avg FPS: {avg_fps:.2f}")
        
        logger.info("Camera stream stopped successfully")
    
    def is_running(self) -> bool:
        """
        Check if the camera stream is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        return self.running
    
    def get_frame_info(self) -> Dict[str, Any]:
        """
        Get information about the current frame and stream status.
        
        Returns:
            Dict containing frame information
        """
        with self.lock:
            info = {
                'running': self.running,
                'frame_count': self.frame_count,
                'last_frame_time': self.last_frame_time,
                'has_frame': self.frame is not None
            }
            
            if self.frame is not None:
                info.update({
                    'frame_shape': self.frame.shape,
                    'frame_dtype': str(self.frame.dtype)
                })
            
            if self.start_time:
                info['uptime'] = time.time() - self.start_time
            
            return info
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Configuration
    config = {
        'width': 1280,
        'height': 720,
        'fps': 30
    }
    
    # Test the camera stream
    camera = CameraStream(camera_id=0, config=config)
    
    try:
        # Start camera
        if camera.start():
            print("Camera started successfully")
            
            # Run for a short test period
            test_duration = 10  # seconds
            start_time = time.time()
            
            while time.time() - start_time < test_duration:
                frame = camera.read()
                
                if frame is not None:
                    # Display frame info
                    info = camera.get_frame_info()
                    print(f"Frame: {info['frame_count']}, "
                          f"Shape: {info['frame_shape']}, "
                          f"Uptime: {info['uptime']:.2f}s")
                    
                    # Optional: Display frame (uncomment to show video window)
                    # cv2.imshow('Camera Stream', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    print("No frame available")
                
                time.sleep(0.1)  # 10 FPS display rate
            
            print("Test completed successfully")
        else:
            print("Failed to start camera")
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()