# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import csv
import os
import time
import numpy as np
import threading
import yaml

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil

# Shared state for continuous motor control
_motor_state = {}
_motor_state_lock = threading.Lock()
_motor_control_running = False
_motor_buses = {}  # Dictionary mapping bus channels to bus objects
_motor_to_bus = {}  # Dictionary mapping motor_id to bus channel
_motor_configs = {}
_motor_current_targets = {}
_motor_configs_lock = threading.Lock()

# Default motor-to-bus mapping (can be overridden)
# Based on leg configuration: can0 = left leg, can1 = right leg
DEFAULT_MOTOR_BUS_MAPPING = {
    # can0 (left leg): hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
    1: "can0", 3: "can0", 5: "can0", 7: "can0", 11: "can0", 13: "can0",
    # can1 (right leg): hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
    2: "can1", 4: "can1", 6: "can1", 8: "can1", 12: "can1", 14: "can1",
}

# Joint to motor ID mapping for legs
# Left leg joints: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
LEFT_LEG_MOTOR_IDS = [1, 3, 5, 7, 11, 13]
# Right leg joints: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
RIGHT_LEG_MOTOR_IDS = [2, 4, 6, 8, 12, 14]

# Calibration offsets mapping: motor_id -> offset_index in calibration.yaml
# First 6 values (indices 0-5) are for left leg motors: 1, 3, 5, 7, 11, 13
# Last 6 values (indices 6-11) are for right leg motors: 2, 4, 6, 8, 12, 14
CALIBRATION_OFFSET_INDEX = {
    1: 0,   # left hip_roll
    3: 1,   # left hip_yaw
    5: 2,   # left hip_pitch
    7: 3,   # left knee_pitch
    11: 4,  # left ankle_pitch
    13: 5,  # left ankle_roll
    2: 6,   # right hip_roll
    4: 7,   # right hip_yaw
    6: 8,   # right hip_pitch
    8: 9,   # right knee_pitch
    12: 10, # right ankle_pitch
    14: 11, # right ankle_roll
}

# Global calibration offsets (loaded from calibration.yaml)
_calibration_offsets = None
_calibration_loaded = False


def move_actuator(motors, iterations, index):
    """
    Utility function to move multiple leg actuators to target positions.
    
    Args:
        motors: List or tuple of (motor_id, target_angle) tuples
               Example: [(1, 1.5), (2, 0.8)] or ((1, 1.5), (2, 0.8))
    """
    args = recoil.util.get_args()
    
    # Create buses for both can0 and can1
    buses = {}
    unique_channels = set()
    for motor_id, _ in motors:
        bus_channel = DEFAULT_MOTOR_BUS_MAPPING.get(motor_id, "can1")
        unique_channels.add(bus_channel)
    
    for channel in unique_channels:
        buses[channel] = recoil.Bus(channel=channel, bitrate=1000000)
    
    # Initialize all motors
    motor_configs = {}
    start_angles = {}
    
    for motor_id, target in motors:
        device_id = motor_id
        bus_channel = DEFAULT_MOTOR_BUS_MAPPING.get(device_id, "can1")
        bus = buses[bus_channel]
        
        # Get motor configuration
        config = _initialize_motor(device_id, bus, target)
        
        motor_configs[device_id] = {
            'kp': config['kp'],
            'kd': config['kd'],
            'torque_limit': config['torque_limit'],
            'target': target,
            'bus': bus
        }
        
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
                bus = config['bus']
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
                    bus = config['bus']
                    measured_position, measured_velocity = bus.write_read_pdo_2(device_id, target_angle, 0.0)
                    if measured_position is not None and measured_velocity is not None:
                        print(f"Motor {device_id} - Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f} \t Target: {target_angle:.3f}")
                
                start_percentage += 0.01
                if start_percentage > 1.0:
                    start_percentage = 1.0
                
                rate.sleep()

        except KeyboardInterrupt:
            pass
    
    # Set all motors to IDLE mode and stop all buses
    for device_id, config in motor_configs.items():
        bus = config['bus']
        bus.set_mode(device_id, recoil.Mode.IDLE)
    
    for bus in buses.values():
        bus.stop()


def _initialize_motor(device_id, bus, initial_target=0.0):
    """
    Initialize a single leg motor with appropriate settings.
    
    Args:
        device_id: Motor device ID
        bus: CAN bus object
        initial_target: Initial target angle in radians
    
    Returns:
        dict: Motor configuration
    """
    # Leg motor configuration
    # Hip roll and hip yaw (motors 1, 2, 3, 4) - higher torque for stability
    if device_id in [1, 2, 3, 4]:
        kp = 50.0
        kd = 2.0
        torque_limit = 6.0
    # Hip pitch, knee pitch, ankle pitch (motors 5, 6, 7, 8, 11, 12) - standard settings
    elif device_id in [5, 6, 7, 8, 11, 12]:
        kp = 20.0
        kd = 4.0
        torque_limit = 4.0
    # Ankle roll (motors 13, 14) - lower torque for fine control
    elif device_id in [13, 14]:
        kp = 20.0
        kd = 5.0
        torque_limit = 2.5
    else:
        # Default settings for any other motor IDs
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
        'torque_limit': torque_limit,
        'bus': bus  # Store bus reference
    }


def update_motor_angles(motors):
    """
    Update the target angles for leg motors. This function can be called to update
    motor targets in real-time. Motors will be automatically initialized if they
    haven't been initialized yet.
    
    Args:
        motors: List or tuple of (motor_id, target_angle) tuples in radians
               Example: [(1, 1.5), (2, 0.8), (5, 0.5)] - leg motor IDs
    """
    global _motor_state, _motor_state_lock
    
    with _motor_state_lock:
        for motor_id, target_angle in motors:
            _motor_state[motor_id] = target_angle


def load_calibration_offsets(calibration_path=None):
    """
    Load calibration offsets from calibration.yaml file.
    
    Args:
        calibration_path: Path to calibration.yaml file. If None, searches in common locations.
    
    Returns:
        dict: Dictionary mapping motor_id to calibration offset
    """
    global _calibration_offsets, _calibration_loaded
    
    if _calibration_loaded and _calibration_offsets is not None:
        return _calibration_offsets
    
    if calibration_path is None:
        # Try common locations
        possible_paths = [
            "calibration.yaml",
            os.path.join("source", "berkeley_humanoid_lite_lowlevel", "scripts", "calibration.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "source", "berkeley_humanoid_lite_lowlevel", "scripts", "calibration.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                calibration_path = path
                break
        
        if calibration_path is None:
            print("[WARNING] Calibration file not found. Motor angles will not be calibrated.")
            _calibration_offsets = {}
            _calibration_loaded = True
            return _calibration_offsets
    
    if not os.path.exists(calibration_path):
        print(f"[WARNING] Calibration file not found at {calibration_path}. Motor angles will not be calibrated.")
        _calibration_offsets = {}
        _calibration_loaded = True
        return _calibration_offsets
    
    try:
        with open(calibration_path, 'r') as f:
            config = yaml.safe_load(f)
        
        position_offsets = config.get("position_offsets", [])
        
        if len(position_offsets) != 12:
            print(f"[WARNING] Expected 12 calibration offsets, got {len(position_offsets)}. Motor angles will not be calibrated.")
            _calibration_offsets = {}
            _calibration_loaded = True
            return _calibration_offsets
        
        # Map offsets to motor IDs
        _calibration_offsets = {}
        for motor_id, offset_index in CALIBRATION_OFFSET_INDEX.items():
            _calibration_offsets[motor_id] = float(position_offsets[offset_index])
        
        print(f"[INFO] Loaded calibration offsets from {calibration_path}")
        _calibration_loaded = True
        return _calibration_offsets
    
    except Exception as e:
        print(f"[WARNING] Error loading calibration file: {e}. Motor angles will not be calibrated.")
        _calibration_offsets = {}
        _calibration_loaded = True
        return _calibration_offsets


def apply_calibration_offset(motor_id, angle):
    """
    Apply calibration offset to a motor angle.
    
    Args:
        motor_id: Motor ID
        angle: Raw angle from CSV
    
    Returns:
        float: Angle with calibration offset applied
    """
    offsets = load_calibration_offsets()
    offset = offsets.get(motor_id, 0.0)
    return angle + offset


def load_joint_angles_from_csv(csv_path, row_index=0, apply_calibration=True):
    """
    Load joint angles from a CSV file recorded by play_mujoco.py.
    
    Args:
        csv_path: Path to the CSV file
        row_index: Index of the row to read (0 = first data row, after header)
        apply_calibration: If True, apply calibration offsets from calibration.yaml
    
    Returns:
        dict: Dictionary mapping motor_id to target angle (with calibration applied if requested)
              Example: {1: 0.1, 3: 0.2, 5: -0.3, ...}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    motor_targets = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if row_index >= len(rows):
            raise IndexError(f"Row index {row_index} out of range. CSV has {len(rows)} data rows.")
        
        row = rows[row_index]
        
        # Map left leg joints to motor IDs
        motor_targets[1] = float(row['leg_left_hip_roll_joint'])      # Motor 1
        motor_targets[3] = float(row['leg_left_hip_yaw_joint'])       # Motor 3
        motor_targets[5] = float(row['leg_left_hip_pitch_joint'])      # Motor 5
        motor_targets[7] = float(row['leg_left_knee_pitch_joint'])   # Motor 7
        motor_targets[11] = float(row['leg_left_ankle_pitch_joint'])  # Motor 11
        motor_targets[13] = float(row['leg_left_ankle_roll_joint'])   # Motor 13
        
        # Map right leg joints to motor IDs
        motor_targets[2] = float(row['leg_right_hip_roll_joint'])      # Motor 2
        motor_targets[4] = float(row['leg_right_hip_yaw_joint'])       # Motor 4
        motor_targets[6] = float(row['leg_right_hip_pitch_joint'])    # Motor 6
        motor_targets[8] = float(row['leg_right_knee_pitch_joint'])   # Motor 8
        motor_targets[12] = float(row['leg_right_ankle_pitch_joint']) # Motor 12
        motor_targets[14] = float(row['leg_right_ankle_roll_joint'])   # Motor 14
    
    # Apply calibration offsets if requested
    if apply_calibration:
        for motor_id in motor_targets:
            motor_targets[motor_id] = apply_calibration_offset(motor_id, motor_targets[motor_id])
    
    return motor_targets


def load_all_joint_angles_from_csv(csv_path, apply_calibration=True):
    """
    Load all joint angle sequences from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        apply_calibration: If True, apply calibration offsets from calibration.yaml
    
    Returns:
        list: List of dictionaries, each mapping motor_id to target angle (with calibration applied if requested)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    all_sequences = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            motor_targets = {}
            
            # Map left leg joints to motor IDs
            motor_targets[1] = float(row['leg_left_hip_roll_joint'])
            motor_targets[3] = float(row['leg_left_hip_yaw_joint'])
            motor_targets[5] = float(row['leg_left_hip_pitch_joint'])
            motor_targets[7] = float(row['leg_left_knee_pitch_joint'])
            motor_targets[11] = float(row['leg_left_ankle_pitch_joint'])
            motor_targets[13] = float(row['leg_left_ankle_roll_joint'])
            
            # Map right leg joints to motor IDs
            motor_targets[2] = float(row['leg_right_hip_roll_joint'])
            motor_targets[4] = float(row['leg_right_hip_yaw_joint'])
            motor_targets[6] = float(row['leg_right_hip_pitch_joint'])
            motor_targets[8] = float(row['leg_right_knee_pitch_joint'])
            motor_targets[12] = float(row['leg_right_ankle_pitch_joint'])
            motor_targets[14] = float(row['leg_right_ankle_roll_joint'])
            
            # Apply calibration offsets if requested
            if apply_calibration:
                for motor_id in motor_targets:
                    motor_targets[motor_id] = apply_calibration_offset(motor_id, motor_targets[motor_id])
            
            all_sequences.append(motor_targets)
    
    return all_sequences


def start_continuous_motor_control(initial_motors=None, update_interval_ms=50, motor_bus_mapping=None):
    """
    Start continuous leg motor control that updates every update_interval_ms milliseconds.
    This function runs in a separate thread and continuously sends commands to motors.
    Use update_motor_angles() to update target angles.
    Motors can be added dynamically - they will be initialized automatically.
    
    Args:
        initial_motors: List or tuple of (motor_id, target_angle) tuples in radians
                       Example: [(1, 1.5), (2, 0.8)] or None to start with no motors
        update_interval_ms: Update interval in milliseconds (default: 50ms)
        motor_bus_mapping: Dictionary mapping motor_id to bus channel
                          Example: {1: "can0", 3: "can0", 2: "can1", 4: "can1"}
                          If None, uses DEFAULT_MOTOR_BUS_MAPPING
    
    Returns:
        Thread object that can be used to stop the control loop
    """
    global _motor_state, _motor_state_lock, _motor_control_running
    global _motor_buses, _motor_to_bus, _motor_configs, _motor_current_targets, _motor_configs_lock
    
    # Use provided mapping or default
    if motor_bus_mapping is None:
        motor_bus_mapping = DEFAULT_MOTOR_BUS_MAPPING.copy()
    
    _motor_to_bus = motor_bus_mapping.copy()
    
    # Create buses for each unique bus channel needed
    args = recoil.util.get_args()
    unique_buses = set(motor_bus_mapping.values())
    for bus_channel in unique_buses:
        if bus_channel not in _motor_buses:
            bus = recoil.Bus(channel=bus_channel, bitrate=1000000)
            _motor_buses[bus_channel] = bus
            print(f"Created bus connection: {bus_channel}")
    
    # Initialize shared state with initial angles
    if initial_motors:
        with _motor_state_lock:
            for motor_id, target_angle in initial_motors:
                _motor_state[motor_id] = target_angle
        
        # Initialize all motors
        with _motor_configs_lock:
            for motor_id, target in initial_motors:
                device_id = motor_id
                # Get the correct bus for this motor
                bus_channel = _motor_to_bus.get(device_id, "can1")  # Default to can1 if not mapped
                if bus_channel not in _motor_buses:
                    # Create bus if it doesn't exist
                    bus_obj = recoil.Bus(channel=bus_channel, bitrate=1000000)
                    _motor_buses[bus_channel] = bus_obj
                    print(f"Created bus connection: {bus_channel}")
                else:
                    bus_obj = _motor_buses[bus_channel]
                
                config = _initialize_motor(device_id, bus_obj, target)
                _motor_configs[device_id] = config
                _motor_current_targets[device_id] = target
    
    update_interval = update_interval_ms / 1000.0  # Convert to seconds
    rate = RateLimiter(frequency=1.0 / update_interval)
    
    def control_loop():
        global _motor_state, _motor_state_lock, _motor_control_running
        global _motor_buses, _motor_to_bus, _motor_configs, _motor_current_targets, _motor_configs_lock
        
        _motor_control_running = True
        
        try:
            while _motor_control_running:
                # Get current target angles from shared state
                with _motor_state_lock:
                    targets = _motor_state.copy()
                
                # Check for new motors that need to be initialized
                with _motor_configs_lock:
                    for device_id in targets.keys():
                        if device_id not in _motor_configs:
                            # Get the correct bus for this motor
                            bus_channel = _motor_to_bus.get(device_id, "can1")  # Default to can1 if not mapped
                            
                            # Create bus if it doesn't exist
                            if bus_channel not in _motor_buses:
                                bus_obj = recoil.Bus(channel=bus_channel, bitrate=1000000)
                                _motor_buses[bus_channel] = bus_obj
                                print(f"Created bus connection: {bus_channel}")
                            else:
                                bus_obj = _motor_buses[bus_channel]
                            
                            # Initialize new motor dynamically
                            print(f"Initializing new motor {device_id} on {bus_channel}")
                            config = _initialize_motor(device_id, bus_obj, targets[device_id])
                            _motor_configs[device_id] = config
                            _motor_current_targets[device_id] = targets[device_id]
                
                # Update current targets and send commands
                with _motor_configs_lock:
                    for device_id in list(_motor_configs.keys()):
                        if device_id in targets:
                            # Update target angle
                            _motor_current_targets[device_id] = targets[device_id]
                        
                        # Get the correct bus for this motor
                        config = _motor_configs[device_id]
                        bus = config.get('bus')
                        
                        if bus is not None:
                            # Send command to motor
                            target_angle = _motor_current_targets[device_id]
                            measured_position, measured_velocity = bus.write_read_pdo_2(device_id, target_angle, 0.0)
                            
                            if measured_position is not None and measured_velocity is not None:
                                print(f"Motor {device_id} - Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f} \t Target: {target_angle:.3f}")
                
                rate.sleep()
        
        except Exception as e:
            print(f"Error in motor control loop: {e}")
        finally:
            # Set all motors to IDLE mode and stop all buses
            with _motor_configs_lock:
                for device_id in _motor_configs.keys():
                    config = _motor_configs[device_id]
                    bus = config.get('bus')
                    if bus is not None:
                        try:
                            bus.set_mode(device_id, recoil.Mode.IDLE)
                        except:
                            pass
                
                # Stop all buses
                for bus_channel, bus in _motor_buses.items():
                    try:
                        bus.stop()
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
    global _motor_control_running, _motor_buses, _motor_configs, _motor_configs_lock
    _motor_control_running = False
    
    # Set all motors to IDLE and stop all buses
    with _motor_configs_lock:
        for device_id in _motor_configs.keys():
            config = _motor_configs[device_id]
            bus = config.get('bus')
            if bus is not None:
                try:
                    bus.set_mode(device_id, recoil.Mode.IDLE)
                except:
                    pass
        
        # Stop all buses
        for bus_channel, bus in _motor_buses.items():
            try:
                bus.stop()
            except:
                pass


# Example usage:
# 
# # Load joint angles from CSV file
# csv_path = "actions_record_20251126_133201.csv"
# 
# # Option 1: Load a specific row from CSV
# motor_targets = load_joint_angles_from_csv(csv_path, row_index=0)
# motors_list = [(motor_id, angle) for motor_id, angle in motor_targets.items()]
# 
# # Option 2: Start continuous control with initial positions from CSV
# motor_targets = load_joint_angles_from_csv(csv_path, row_index=0)
# motors_list = [(motor_id, angle) for motor_id, angle in motor_targets.items()]
# control_thread = start_continuous_motor_control(initial_motors=motors_list)
# 
# # Option 3: Play back entire sequence from CSV
# all_sequences = load_all_joint_angles_from_csv(csv_path)
# for i, motor_targets in enumerate(all_sequences):
#     motors_list = [(motor_id, angle) for motor_id, angle in motor_targets.items()]
#     update_motor_angles(motors_list)
#     time.sleep(0.04)  # Match the control frequency (25 Hz)
# 
# # Stop control when done
# stop_continuous_motor_control()