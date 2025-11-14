# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import numpy as np
import threading

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil

# Shared state for continuous motor control
_motor_state = {}
_motor_state_lock = threading.Lock()
_motor_control_running = False
_motor_bus = None
_motor_configs = {}
_motor_current_targets = {}
_motor_configs_lock = threading.Lock()


def move_actuator(motors, iterations, index):
    """
    Utility function to move multiple actuators to target positions.
    
    Args:
        motors: List or tuple of (motor_id, target_angle) tuples
               Example: [(2, 1.5), (4, 0.8)] or ((2, 1.5), (4, 0.8))
    """
    args = recoil.util.get_args()
    bus = recoil.Bus(channel="can1", bitrate=1000000)
    
    # Initialize all motors
    motor_configs = {}
    start_angles = {}
    
    for motor_id, target in motors:
        device_id = motor_id
        
        if device_id == 2 or device_id == 4:
            kp = 50.0
            kd = 2.0
            torque_limit = 6.0
        else:
            kp = 20.0
            kd = 4.0
            torque_limit = 4.0
        
        motor_configs[device_id] = {
            'kp': kp,
            'kd': kd,
            'torque_limit': torque_limit,
            'target': target
        }
        
        bus.write_position_kp(device_id, kp)
        bus.write_position_kd(device_id, kd)
        bus.write_torque_limit(device_id, torque_limit)
        bus.write_gear_ratio(device_id, -15.0)
        bus.write_position_limit_upper(device_id, np.inf)
        bus.write_position_limit_lower(device_id, -np.inf)
        
        bus.set_mode(device_id, recoil.Mode.POSITION)
        bus.feed(device_id)
        
        start_angles[device_id] = bus.read_position_measured(device_id) or 0.0
    
    frequency = 1.0  # motion frequency is 1 Hz
    amplitude = 1.0  # motion amplitude is 1 rad
    
    rate = RateLimiter(frequency=200.0)
    
    start_percentage = 0.0
    start_time = time.time()
    duration = 2.0  # Run for 2 second
    if index < iterations - 1:
        while time.time() - start_time < duration:
            for device_id, config in motor_configs.items():
                target_angle = (start_percentage * config['target'] + (1 - start_percentage) * start_angles[device_id])
                measured_position, measured_velocity = bus.write_read_pdo_2(device_id, target_angle, 0.0)
                if measured_position is not None and measured_velocity is not None:
                    print(f"Motor {device_id} - Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f} \t Target: {target_angle:.3f}")
            
            start_percentage += 0.01
            if start_percentage > 1.0:
                start_percentage = 1.0
            
            rate.sleep()
    else:
        try:
            while True:
                for device_id, config in motor_configs.items():
                    target_angle = (start_percentage * config['target'] + (1 - start_percentage) * start_angles[device_id])
                    measured_position, measured_velocity = bus.write_read_pdo_2(device_id, target_angle, 0.0)
                    if measured_position is not None and measured_velocity is not None:
                        print(f"Motor {device_id} - Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f} \t Target: {target_angle:.3f}")
                
                start_percentage += 0.01
                if start_percentage > 1.0:
                    start_percentage = 1.0
                
                rate.sleep()

        except KeyboardInterrupt:
            pass
    
    # Set all motors to IDLE mode
    for device_id in motor_configs.keys():
        bus.set_mode(device_id, recoil.Mode.IDLE)
    
    bus.stop()


def _initialize_motor(device_id, bus, initial_target=0.0):
    """
    Initialize a single motor with appropriate settings.
    
    Args:
        device_id: Motor device ID
        bus: CAN bus object
        initial_target: Initial target angle in radians
    
    Returns:
        dict: Motor configuration
    """
    if device_id == 2 or device_id == 4:
        kp = 50.0
        kd = 2.0
        torque_limit = 6.0
    else:
        kp = 20.0
        kd = 4.0
        torque_limit = 4.0
    
    bus.write_position_kp(device_id, kp)
    bus.write_position_kd(device_id, kd)
    bus.write_torque_limit(device_id, torque_limit)
    bus.write_gear_ratio(device_id, -15.0)
    bus.write_position_limit_upper(device_id, np.inf)
    bus.write_position_limit_lower(device_id, -np.inf)
    
    bus.set_mode(device_id, recoil.Mode.POSITION)
    bus.feed(device_id)
    
    return {
        'kp': kp,
        'kd': kd,
        'torque_limit': torque_limit
    }


def update_motor_angles(motors):
    """
    Update the target angles for motors. This function can be called from vision logic
    to update motor targets in real-time. Motors will be automatically initialized
    if they haven't been initialized yet.
    
    Args:
        motors: List or tuple of (motor_id, target_angle) tuples in radians
               Example: [(2, 1.5), (8, 0.8), (1, 0.5)] - can be any motor IDs
    """
    global _motor_state, _motor_state_lock
    
    with _motor_state_lock:
        for motor_id, target_angle in motors:
            _motor_state[motor_id] = target_angle


def start_continuous_motor_control(initial_motors=None, update_interval_ms=50, bus_channel="can1"):
    """
    Start continuous motor control that updates every update_interval_ms milliseconds.
    This function runs in a separate thread and continuously sends commands to motors.
    Use update_motor_angles() to update target angles from your vision logic.
    Motors can be added dynamically - they will be initialized automatically.
    
    Args:
        initial_motors: List or tuple of (motor_id, target_angle) tuples in radians
                       Example: [(2, 1.5), (8, 0.8)] or None to start with no motors
        update_interval_ms: Update interval in milliseconds (default: 50ms)
        bus_channel: CAN bus channel (default: "can1")
    
    Returns:
        Thread object that can be used to stop the control loop
    """
    global _motor_state, _motor_state_lock, _motor_control_running
    global _motor_bus, _motor_configs, _motor_current_targets, _motor_configs_lock
    
    args = recoil.util.get_args()
    bus = recoil.Bus(channel=bus_channel, bitrate=1000000)
    _motor_bus = bus
    
    # Initialize shared state with initial angles
    if initial_motors:
        with _motor_state_lock:
            for motor_id, target_angle in initial_motors:
                _motor_state[motor_id] = target_angle
        
        # Initialize all motors
        with _motor_configs_lock:
            for motor_id, target in initial_motors:
                device_id = motor_id
                config = _initialize_motor(device_id, bus, target)
                _motor_configs[device_id] = config
                _motor_current_targets[device_id] = target
    
    update_interval = update_interval_ms / 1000.0  # Convert to seconds
    rate = RateLimiter(frequency=1.0 / update_interval)
    
    def control_loop():
        global _motor_state, _motor_state_lock, _motor_control_running
        global _motor_bus, _motor_configs, _motor_current_targets, _motor_configs_lock
        
        _motor_control_running = True
        
        try:
            while _motor_control_running:
                # Get current target angles from shared state
                with _motor_state_lock:
                    targets = _motor_state.copy()
                
                # Check for new motors that need to be initialized
                if _motor_bus is not None:
                    with _motor_configs_lock:
                        for device_id in targets.keys():
                            if device_id not in _motor_configs:
                                # Initialize new motor dynamically
                                print(f"Initializing new motor {device_id}")
                                config = _initialize_motor(device_id, _motor_bus, targets[device_id])
                                _motor_configs[device_id] = config
                                _motor_current_targets[device_id] = targets[device_id]
                
                # Update current targets and send commands
                if _motor_bus is not None:
                    with _motor_configs_lock:
                        for device_id in list(_motor_configs.keys()):
                            if device_id in targets:
                                # Update target angle
                                _motor_current_targets[device_id] = targets[device_id]
                            
                            # Send command to motor
                            target_angle = _motor_current_targets[device_id]
                            measured_position, measured_velocity = _motor_bus.write_read_pdo_2(device_id, target_angle, 0.0)
                            
                            if measured_position is not None and measured_velocity is not None:
                                print(f"Motor {device_id} - Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f} \t Target: {target_angle:.3f}")
                
                rate.sleep()
        
        except Exception as e:
            print(f"Error in motor control loop: {e}")
        finally:
            # Set all motors to IDLE mode
            if _motor_bus is not None:
                with _motor_configs_lock:
                    for device_id in _motor_configs.keys():
                        try:
                            _motor_bus.set_mode(device_id, recoil.Mode.IDLE)
                        except:
                            pass
                    try:
                        _motor_bus.stop()
                    except:
                        pass
            _motor_control_running = False
    
    # Start control loop in a separate thread
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()
    
    return control_thread


def stop_continuous_motor_control():
    """
    Stop the continuous motor control loop.
    """
    global _motor_control_running
    _motor_control_running = False
    if _motor_bus is not None:
        with _motor_configs_lock:
            for device_id in _motor_configs.keys():
                _motor_bus.set_mode(device_id, recoil.Mode.IDLE)
            _motor_bus.stop()
                
                    
                

