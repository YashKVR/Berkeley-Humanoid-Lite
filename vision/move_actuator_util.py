# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import numpy as np

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil


def move_actuator(motor_id, target):
    """
    Utility function to move an actuator to a target position.
    
    Args:
        target: Target position in radians
    """
    args = recoil.util.get_args()
    bus = recoil.Bus(channel=args.channel, bitrate=1000000)

    device_id = motor_id

    if device_id == 2 or device_id == 4:
        kp=50.0
        kd=2.0
        torque_limit=6.0
    else:
        kp=20.0
        kd=4.0
        torque_limit=4.0

    frequency = 1.0  # motion frequency is 1 Hz
    amplitude = 1.0  # motion amplitude is 1 rad

    rate = RateLimiter(frequency=200.0)

    bus.write_position_kp(device_id, kp)
    bus.write_position_kd(device_id, kd)
    bus.write_torque_limit(device_id, torque_limit)
    bus.write_gear_ratio(device_id, -15.0)
    bus.write_position_limit_upper(device_id, np.inf)
    bus.write_position_limit_lower(device_id, -np.inf)

    bus.set_mode(device_id, recoil.Mode.POSITION)
    bus.feed(device_id)

    start_percentage = 0.0
    start_angle = bus.read_position_measured(device_id) or 0.0
    

    try:
        while True:
            # target_angle = np.sin(2 * np.pi * frequency * time.time()) * amplitude
            target_angle = start_angle + (start_percentage * target + (1 - start_percentage) * start_angle)
            start_percentage += 0.01
            if start_percentage > 1.0:
                start_percentage = 1.0
            measured_position, measured_velocity = bus.write_read_pdo_2(device_id, target_angle, 0.0)
            if measured_position is not None and measured_velocity is not None:
                print(f"Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f}")

            rate.sleep()

    except KeyboardInterrupt:
        pass

    bus.set_mode(device_id, recoil.Mode.IDLE)
    bus.stop()

