# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import numpy as np

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil


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

