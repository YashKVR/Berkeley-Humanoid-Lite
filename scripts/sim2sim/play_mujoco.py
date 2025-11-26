# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.


import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg


# Load configuration
cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")


# Main execution block
def main():
    """Main execution function for the MuJoCo simulation environment."""
    # Initialize environment
    robot = MujocoSimulator(cfg)
    obs = robot.reset()

    # Initialize and start policy controller
    controller = RlController(cfg)
    controller.load_policy()

    # Default actions for fallback
    default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    # Setup CSV recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"actions_record_{timestamp}.csv"
    
    # Get joint names from config
    joint_names = cfg.joints if hasattr(cfg, 'joints') else [f"joint_{i}" for i in range(cfg.num_actions)]
    # Filter to only action indices if needed
    if hasattr(cfg, 'action_indices'):
        joint_names = [joint_names[i] for i in cfg.action_indices]
    
    # Create CSV file with headers
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp'] + joint_names)
    
    print(f"[INFO] Recording actions to {csv_filename}")
    
    try:
        # Main control loop
        while True:
            # Send observations and receive actions
            actions = controller.update(obs.numpy())

            # Use default actions if no actions received
            if actions is None:
                actions = default_actions

            # Record actions to CSV
            timestamp = time.time()
            # csv_writer.writerow([timestamp] + actions.tolist() if isinstance(actions, np.ndarray) else [timestamp] + list(actions))

            # Execute step
            actions = torch.tensor(actions)
            obs = robot.step(actions)
    
    except KeyboardInterrupt:
        print(f"\n[INFO] Stopping recording. Actions saved to {csv_filename}")
    finally:
        csv_file.close()
        print(f"[INFO] CSV file closed: {csv_filename}")


if __name__ == "__main__":
    main()
