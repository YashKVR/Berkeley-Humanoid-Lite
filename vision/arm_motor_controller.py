#!/usr/bin/env python
#
# Arm Motor Controller
# 
# Controls right arm motors using move_actuator-style logic with dynamic target updates.
# Updates target angles every timestep based on MediaPipe feed.
#

import time
import numpy as np
import math
import logging
from typing import Dict, Optional
from threading import Thread, Lock
import signal

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil

logger = logging.getLogger(__name__)


class ArmMotorController:
    """
    Motor controller for right arm using move_actuator-style logic.
    
    Allows dynamic target updates while smoothly interpolating to new targets.
    """
    
    # Right arm joint to motor device ID mapping
    # Note: All motors communicate on the same CAN bus channel (can0)
    # These are the device IDs (motor IDs) on that bus, not CAN channels
    JOINT_TO_MOTOR_ID = {
        'arm_right_shoulder_pitch_joint': 2,
        'arm_right_shoulder_roll_joint': 4,
        'arm_right_shoulder_yaw_joint': 6,
        'arm_right_elbow_pitch_joint': 8,
        'arm_right_elbow_roll_joint': 10,
    }
    
    def __init__(
        self,
        can_channel: str = 'can0',
        control_rate: float = 200.0,  # Hz (same as move_actuator)
        interpolation_rate: float = 0.05  # How fast to interpolate (0.05 = 5% per step at 200Hz = ~0.2 seconds to target)
    ):
        """
        Initialize arm motor controller.
        
        Args:
            can_channel: CAN bus channel (e.g., 'can0')
            control_rate: Control loop frequency in Hz
            interpolation_rate: Interpolation rate per control step (0.01 = 1% per step)
        """
        self.can_channel = can_channel
        self.control_rate = control_rate
        self.interpolation_rate = interpolation_rate
        
        # Motor control state
        self.bus: Optional[recoil.Bus] = None
        self.running = False
        self.control_thread: Optional[Thread] = None
        
        # Target angles (in radians) - updated by external code
        self.target_angles: Dict[str, Optional[float]] = {}
        self.current_angles: Dict[str, float] = {}  # Current interpolated angles
        self.start_angles: Dict[str, float] = {}  # Starting angles when target changes
        self.interpolation_progress: Dict[str, float] = {}  # 0.0 to 1.0
        
        # Track motor responsiveness (skip motors that consistently don't respond)
        self.motor_responsive: Dict[str, bool] = {}  # Track if motor is responding
        self.motor_failure_count: Dict[str, int] = {}  # Count consecutive failures
        self.motor_success_count: Dict[str, int] = {}  # Count consecutive successes
        
        # Thread safety
        self.lock = Lock()
        
        # Initialize target angles dict
        for joint_name in self.JOINT_TO_MOTOR_ID.keys():
            self.target_angles[joint_name] = None
            self.current_angles[joint_name] = 0.0
            self.start_angles[joint_name] = 0.0
            self.interpolation_progress[joint_name] = 1.0  # Start at 1.0 (no interpolation needed)
            self.motor_responsive[joint_name] = True  # Assume responsive until proven otherwise
            self.motor_failure_count[joint_name] = 0
            self.motor_success_count[joint_name] = 0
    
    def _get_motor_params(self, motor_id: int) -> tuple:
        """Get motor parameters (kp, kd, torque_limit) based on motor device ID."""
        if motor_id == 2 or motor_id == 4:
            return 50.0, 2.0, 6.0
        else:
            return 20.0, 4.0, 4.0
    
    def initialize_motors(self) -> bool:
        """
        Initialize all right arm motors.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Use the can_channel directly (don't rely on command line args)
            # This avoids issues with get_args() blocking or failing
            channel = self.can_channel
            logger.info(f"Initializing CAN bus on channel: {channel}")
            
            self.bus = recoil.Bus(channel=channel, bitrate=1000000)
            
            # Initialize each motor
            if not self.bus:
                logger.error("Bus not initialized")
                return False
                
            for joint_name, motor_id in self.JOINT_TO_MOTOR_ID.items():
                try:
                    kp, kd, torque_limit = self._get_motor_params(motor_id)
                    
                    # Set motor parameters (motor_id is the device ID on the CAN bus)
                    self.bus.write_position_kp(motor_id, kp)
                    self.bus.write_position_kd(motor_id, kd)
                    self.bus.write_torque_limit(motor_id, torque_limit)
                    self.bus.write_gear_ratio(motor_id, -15.0)
                    self.bus.write_position_limit_upper(motor_id, np.inf)
                    self.bus.write_position_limit_lower(motor_id, -np.inf)
                    
                    # Set mode to POSITION and feed
                    self.bus.set_mode(motor_id, recoil.Mode.POSITION)
                    self.bus.feed(motor_id)
                    time.sleep(0.005)  # Small delay after feed (like move_actuator does)
                    
                    # Skip reading initial position to avoid blocking on unresponsive motors
                    # We'll start from 0.0 and the motors will move to target positions
                    # The actual position will be read during the control loop via write_read_pdo_2
                    initial_pos = None  # Don't read during init to avoid blocking
                    
                    # Initialize with default position (0.0)
                    # The control loop will read actual positions when sending commands
                    with self.lock:
                        self.current_angles[joint_name] = 0.0
                        self.start_angles[joint_name] = 0.0
                        self.target_angles[joint_name] = 0.0
                        self.interpolation_progress[joint_name] = 1.0
                    
                    logger.info(f"✓ Initialized {joint_name} (Motor ID {motor_id} on {self.can_channel}) - "
                              f"kp={kp}, kd={kd}, torque_limit={torque_limit} "
                              f"(position will be read during control loop)")
                    
                    time.sleep(0.001)  # Small delay between motors
                    
                except Exception as e:
                    logger.error(f"✗ Failed to initialize {joint_name} (Motor ID {motor_id} on {self.can_channel}): {e}", exc_info=True)
                    with self.lock:
                        self.current_angles[joint_name] = 0.0
                        self.start_angles[joint_name] = 0.0
                        self.target_angles[joint_name] = 0.0
                        self.interpolation_progress[joint_name] = 1.0
            
            logger.info("Motor initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize motors: {e}", exc_info=True)
            return False
    
    def update_target_angles(self, joint_commands: Dict[str, Optional[float]]):
        """
        Update target angles for joints.
        
        This is called from the main teleop loop with new target angles from MediaPipe.
        
        Args:
            joint_commands: Dictionary mapping joint names to target angles (in radians)
        """
        if not hasattr(self, '_update_counter'):
            self._update_counter = 0
        self._update_counter += 1
        
        with self.lock:
            updated_count = 0
            for joint_name, target_angle in joint_commands.items():
                if joint_name not in self.JOINT_TO_MOTOR_ID:
                    continue
                
                if target_angle is None:
                    continue
                
                # If target changed significantly (more than 0.01 rad ≈ 0.57°), reset interpolation
                # This prevents constant resetting from small noise in pose detection
                current_target = self.target_angles[joint_name]
                if current_target is None or abs(current_target - target_angle) > 0.01:
                    self.start_angles[joint_name] = self.current_angles[joint_name]
                    self.target_angles[joint_name] = target_angle
                    self.interpolation_progress[joint_name] = 0.0
                    updated_count += 1
                elif current_target is not None:
                    # Small change - update target but don't reset interpolation (smooth tracking)
                    self.target_angles[joint_name] = target_angle
            
            # Log updates occasionally (every 100 calls ≈ every 1.6 seconds at 60Hz)
            if self._update_counter % 100 == 0:
                targets_str = ", ".join([f"{name}={math.degrees(angle):.1f}°" if angle is not None else f"{name}=None"
                                        for name, angle in self.target_angles.items()])
                logger.debug(f"update_target_angles (call #{self._update_counter}): Updated {updated_count} joints, targets: [{targets_str}]")
    
    def _control_loop(self):
        """Main control loop running in separate thread."""
        if not self.bus:
            logger.error("Control loop started but bus not initialized")
            return
        
        rate = RateLimiter(frequency=self.control_rate)
        
        logger.info(f"Starting motor control loop at {self.control_rate} Hz")
        
        loop_count = 0
        try:
            logger.info("Control loop running...")
            while self.running:
                loop_count += 1
                if loop_count % 2000 == 0:  # Log every ~10 seconds at 200Hz
                    logger.info(f"Control loop iteration {loop_count}, running={self.running}")
                
                with self.lock:
                    # Update interpolation and send commands for each joint
                    for joint_name, motor_id in self.JOINT_TO_MOTOR_ID.items():
                        target = self.target_angles[joint_name]
                        
                        if target is None:
                            continue
                        
                        # Interpolate towards target
                        progress = self.interpolation_progress[joint_name]
                        if progress < 1.0:
                            # Interpolate: current = start + progress * (target - start)
                            current = self.start_angles[joint_name] + progress * (
                                target - self.start_angles[joint_name]
                            )
                            self.current_angles[joint_name] = current
                            
                            # Update progress
                            progress += self.interpolation_rate
                            if progress > 1.0:
                                progress = 1.0
                            self.interpolation_progress[joint_name] = progress
                        else:
                            # Already at target, use target directly
                            self.current_angles[joint_name] = target
                        
                        # Skip unresponsive motors (after multiple failures)
                        if not self.motor_responsive.get(joint_name, True):
                            continue
                        
                        # Send command to motor (motor_id is the device ID on the CAN bus)
                        try:
                            # Feed motor periodically to keep it active (every 100 iterations = ~0.5 seconds at 200Hz)
                            if loop_count % 100 == 0:
                                self.bus.feed(motor_id)
                            
                            measured_position, measured_velocity = self.bus.write_read_pdo_2(
                                motor_id, self.current_angles[joint_name], 0.0
                            )
                            
                            # Check if motor responded
                            if measured_position is not None:
                                # Motor responded successfully
                                self.motor_success_count[joint_name] = self.motor_success_count.get(joint_name, 0) + 1
                                self.motor_failure_count[joint_name] = 0
                                
                                # Mark as responsive if we had enough successes
                                if self.motor_success_count[joint_name] >= 3:
                                    self.motor_responsive[joint_name] = True
                                
                                # Log success occasionally (every 200 updates = ~1 second at 200Hz)
                                if self.motor_success_count[joint_name] % 200 == 0:
                                    target_deg = math.degrees(self.current_angles[joint_name])
                                    measured_deg = math.degrees(measured_position)
                                    error_deg = abs(target_deg - measured_deg)
                                    logger.info(f"✓ {joint_name} (Motor ID {motor_id}): "
                                              f"target={target_deg:.1f}° "
                                              f"measured={measured_deg:.1f}° "
                                              f"error={error_deg:.1f}° "
                                              f"progress={self.interpolation_progress[joint_name]:.2f}")
                            else:
                                # Motor didn't respond (timeout) - but don't log every time to avoid spam
                                # Only log occasionally
                                self.motor_failure_count[joint_name] = self.motor_failure_count.get(joint_name, 0) + 1
                                self.motor_success_count[joint_name] = 0
                                
                                # Log first few failures and then every 50th failure
                                if self.motor_failure_count[joint_name] <= 3 or self.motor_failure_count[joint_name] % 50 == 0:
                                    logger.debug(f"⚠ {joint_name} (Motor ID {motor_id}) timeout (failure count: {self.motor_failure_count[joint_name]})")
                                
                                # Mark as unresponsive after 10 consecutive failures
                                if self.motor_failure_count[joint_name] >= 10:
                                    if self.motor_responsive.get(joint_name, True):
                                        logger.warning(f"⚠ {joint_name} (Motor ID {motor_id}) not responding - "
                                                      f"skipping future commands. Check motor connection.")
                                        self.motor_responsive[joint_name] = False
                                
                        except Exception as e:
                            # Exception occurred (not just timeout)
                            self.motor_failure_count[joint_name] = self.motor_failure_count.get(joint_name, 0) + 1
                            logger.error(f"✗ Error sending command to {joint_name} (Motor ID {motor_id}): {e}")
                            
                            # Mark as unresponsive after exceptions too
                            if self.motor_failure_count[joint_name] >= 10:
                                self.motor_responsive[joint_name] = False
                
                rate.sleep()
                
        except KeyboardInterrupt:
            logger.info("Control loop interrupted")
        except Exception as e:
            logger.error(f"Error in control loop: {e}", exc_info=True)
        finally:
            logger.info("Control loop stopped")
    
    def start(self):
        """Start the motor control loop in a separate thread."""
        if self.running:
            logger.warning("Motor controller already running")
            return
        
        if not self.bus:
            logger.error("Cannot start: motors not initialized")
            return
        
        self.running = True
        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        logger.info("Motor control loop thread started")
        logger.info(f"Control loop will run at {self.control_rate} Hz")
        logger.info(f"Motors to control: {list(self.JOINT_TO_MOTOR_ID.keys())}")
    
    def stop(self):
        """Stop the motor control loop and set motors to IDLE."""
        if not self.running:
            return
        
        logger.info("Stopping motor control loop...")
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        
        # Set all motors to IDLE
        if self.bus:
            try:
                for joint_name, motor_id in self.JOINT_TO_MOTOR_ID.items():
                    try:
                        self.bus.set_mode(motor_id, recoil.Mode.IDLE)
                    except Exception as e:
                        logger.error(f"Error setting {joint_name} (Motor ID {motor_id}) to IDLE: {e}")
                
                self.bus.stop()
                logger.info("Motors set to IDLE and bus stopped")
            except Exception as e:
                logger.error(f"Error stopping motors: {e}", exc_info=True)
    
    def get_current_angles(self) -> Dict[str, float]:
        """Get current interpolated angles for all joints."""
        with self.lock:
            return self.current_angles.copy()
    
    def get_target_angles(self) -> Dict[str, Optional[float]]:
        """Get target angles for all joints."""
        with self.lock:
            return self.target_angles.copy()

