#!/usr/bin/env python3
"""
UDP Command Listener for Robot
Listens for joint commands sent from the VR teleoperation system.

Usage:
    python udp_command_listener.py [--port PORT] [--host HOST]

The commands are sent in JSON format:
{
    "commands": {
        "xvel": 0,
        "yvel": 0,
        "yawrate": 0,
        "baseheight": 0,
        "baseroll": 0,
        "basepitch": 0,
        "rshoulderpitch": <angle>,
        "rshoulderroll": <angle>,
        "rshoulderyaw": <angle>,
        "relbowpitch": <angle>,
        "rwristroll": <angle>,
        "rgripper": <angle>,
        "lshoulderpitch": <angle>,
        "lshoulderroll": <angle>,
        "lshoulderyaw": <angle>,
        "lelbowpitch": <angle>,
        "lwristroll": <angle>,
        "lgripper": <angle>
    }
}

All joint angles are in radians.
"""

import socket
import json
import argparse
import sys
import os

# Add scripts/vision_to_teleop to path to import move_actuator_util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'vision_to_teleop'))
from move_actuator_util import update_motor_angles, start_continuous_motor_control, stop_continuous_motor_control

def main():
    parser = argparse.ArgumentParser(
        description="Listen for UDP joint commands from VR teleoperation system"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10000,
        help="UDP port to listen on (default: 10000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print received commands to console"
    )
    args = parser.parse_args()

    # Start continuous motor control
    print("Starting continuous motor control...")
    start_continuous_motor_control(update_interval_ms=50)
    print("Motor control started.\n")

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)  # 1 second timeout for graceful shutdown

    print(f"Listening for UDP commands on {args.host}:{args.port}")
    print("Waiting for commands...")
    print("Press Ctrl+C to stop\n")

    message_count = 0
    last_motor_angles = None  # Store last received motor angles to hold position
    
    try:
        while True:
            try:
                # Receive UDP message
                data, addr = sock.recvfrom(4096)  # 4KB buffer
                message_count += 1
                
                # Decode and parse JSON
                try:
                    message_str = data.decode("utf-8").strip()
                    message = json.loads(message_str)
                    
                    if "commands" in message:
                        commands = message["commands"]
                        
                        if args.print:
                            print(f"\n[{message_count}] Received from {addr[0]}:{addr[1]}")
                            print("Commands:")
                            for key, value in commands.items():
                                if isinstance(value, float):
                                    print(f"  {key}: {value:.4f} rad ({value * 57.2958:.2f} deg)")
                                else:
                                    print(f"  {key}: {value}")
                            print()                        
                            
                        # Map commands to motor angles
                        # Store the motor angles mapping for position holding
                        try:
                            # motor_angles = [
                            #     (1, -commands["lshoulderpitch"]),
                            #     (2, -commands["rshoulderpitch"]),
                            #     (3, commands["lshoulderyaw"]),
                            #     (4, commands["rshoulderyaw"]),
                            #     (5, -commands["lshoulderroll"]),
                            #     (6, -commands["rshoulderroll"]),
                            #     (7, -commands["lelbowpitch"]),
                            #     (8, -commands["relbowpitch"])
                            # ]
                            motor_angles = [
                                (1, -commands["lshoulderpitch"]),
                                (2, -commands["rshoulderpitch"]),
                                (3, commands["lshoulderroll"]),
                                (4, commands["rshoulderroll"]),
                                (5, -commands["lshoulderyaw"]),
                                (6, -commands["rshoulderyaw"]),
                                (7, -commands["lelbowpitch"]),
                                (8, -commands["relbowpitch"])
                            ]
                            
                            # Update motor angles with new values
                            update_motor_angles(motor_angles)
                            last_motor_angles = motor_angles  # Store for position holding
                        except KeyError as e:
                            print(f"Warning: Missing required command key: {e}")
                            # Continue with last known angles if available
                            if last_motor_angles is not None:
                                update_motor_angles(last_motor_angles)
                        
                    else:
                        print(f"Warning: Received message without 'commands' key: {message}")
                        
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse JSON: {e}")
                    print(f"Raw data: {data[:100]}...")
                    
            except socket.timeout:
                # Timeout is expected - refresh last known angles to hold position
                # This ensures motors maintain their position even when packets are lost
                if last_motor_angles is not None:
                    update_motor_angles(last_motor_angles)
                continue
                
    except KeyboardInterrupt:
        print(f"\n\nStopped listening. Received {message_count} messages total.")
    finally:
        sock.close()
        print("UDP socket closed")
        print("Stopping motor control...")
        stop_continuous_motor_control()
        print("Motor control stopped.")


if __name__ == "__main__":
    main()

