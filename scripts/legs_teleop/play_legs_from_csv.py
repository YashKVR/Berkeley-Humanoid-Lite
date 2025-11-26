# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Script to play back recorded leg joint angles from CSV file.
This script loads joint angles from a CSV file recorded by play_mujoco.py
and controls the leg motors in real-time.
"""

import argparse
import os
import sys
import time

# Add script directory to path to import move_legs
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from move_legs import (
    load_all_joint_angles_from_csv,
    load_calibration_offsets,
    load_joint_angles_from_csv,
    start_continuous_motor_control,
    stop_continuous_motor_control,
    update_motor_angles,
)


def main():
    """Main function to play back leg joint angles from CSV."""
    parser = argparse.ArgumentParser(
        description="Play back recorded leg joint angles from CSV file"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="actions_record_20251126_133201.csv",
        help="Path to CSV file containing recorded joint angles (default: actions_record_20251126_133201.csv)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Play back only a specific row index (0-based). If not specified, plays back entire sequence.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the playback continuously",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=25.0,
        help="Playback frequency in Hz (default: 25.0 Hz, matching policy control frequency)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Starting row index for playback (default: 0)",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="Ending row index for playback (exclusive). If not specified, plays to end.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable calibration offsets (use raw CSV values)",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to calibration.yaml file (default: auto-detect)",
    )

    args = parser.parse_args()

    # Resolve CSV path - check current directory and script directory
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        # Try current directory first
        if not os.path.exists(csv_path):
            # Try script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, csv_path)
            if not os.path.exists(csv_path):
                # Try parent directory (project root)
                parent_dir = os.path.dirname(script_dir)
                csv_path = os.path.join(parent_dir, args.csv)
                if not os.path.exists(csv_path):
                    print(f"[ERROR] CSV file not found: {args.csv}")
                    print(f"[INFO] Searched in:")
                    print(f"  - Current directory: {os.getcwd()}")
                    print(f"  - Script directory: {script_dir}")
                    print(f"  - Parent directory: {parent_dir}")
                    sys.exit(1)

    print(f"[INFO] Loading CSV file: {csv_path}")

    # Load calibration if not disabled
    apply_calibration = not args.no_calibration
    if apply_calibration:
        if args.calibration_file:
            load_calibration_offsets(calibration_path=args.calibration_file)
        else:
            load_calibration_offsets()
        print("[INFO] Calibration offsets will be applied to motor angles")
    else:
        print("[INFO] Calibration offsets disabled - using raw CSV values")

    try:
        if args.row is not None:
            # Play back single row
            print(f"[INFO] Loading row {args.row} from CSV...")
            motor_targets = load_joint_angles_from_csv(csv_path, row_index=args.row, apply_calibration=apply_calibration)
            motors_list = [(motor_id, angle) for motor_id, angle in motor_targets.items()]

            print(f"[INFO] Starting continuous motor control with row {args.row}...")
            control_thread = start_continuous_motor_control(initial_motors=motors_list)

            print("[INFO] Motor control started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n[INFO] Stopping motor control...")
                stop_continuous_motor_control()
                print("[INFO] Motor control stopped.")

        else:
            # Play back entire sequence
            print("[INFO] Loading all sequences from CSV...")
            all_sequences = load_all_joint_angles_from_csv(csv_path, apply_calibration=apply_calibration)

            # Apply start/end row limits
            start_idx = args.start_row
            end_idx = args.end_row if args.end_row is not None else len(all_sequences)
            sequences = all_sequences[start_idx:end_idx]

            if len(sequences) == 0:
                print(f"[ERROR] No sequences to play (start_row={start_idx}, end_row={end_idx})")
                sys.exit(1)

            print(f"[INFO] Playing back {len(sequences)} sequences (rows {start_idx} to {end_idx-1})")
            print(f"[INFO] Playback frequency: {args.frequency} Hz")
            print("[INFO] Press Ctrl+C to stop playback.")

            # Initialize with first sequence
            initial_motors = [(motor_id, angle) for motor_id, angle in sequences[0].items()]
            control_thread = start_continuous_motor_control(initial_motors=initial_motors)

            sleep_time = 1.0 / args.frequency

            try:
                iteration = 0
                while True:
                    # Update motor angles for current sequence
                    sequence_idx = iteration % len(sequences)
                    motor_targets = sequences[sequence_idx]
                    motors_list = [(motor_id, angle) for motor_id, angle in motor_targets.items()]
                    update_motor_angles(motors_list)

                    if iteration % 100 == 0:
                        print(f"[INFO] Playing sequence {sequence_idx + 1}/{len(sequences)} (iteration {iteration})")

                    time.sleep(sleep_time)
                    iteration += 1

                    # If not looping, stop after one complete cycle
                    if not args.loop and sequence_idx == len(sequences) - 1:
                        print("[INFO] Playback complete.")
                        break

            except KeyboardInterrupt:
                print("\n[INFO] Stopping playback...")
                stop_continuous_motor_control()
                print("[INFO] Playback stopped.")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        stop_continuous_motor_control()
        sys.exit(1)


if __name__ == "__main__":
    main()

