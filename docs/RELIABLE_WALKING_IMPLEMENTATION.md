# Reliable Walking Training Implementation Guide

This document provides a comprehensive guide for implementing a reliable walking training algorithm for the Berkeley Humanoid Lite robot, based on state-of-the-art techniques.

## Overview

The implementation includes:
1. **Enhanced reward functions** for stability, recovery, and energy efficiency
2. **Progressive curriculum learning** for gradual difficulty increase
3. **Improved domain randomization** for robustness
4. **Enhanced observations** with temporal history
5. **New configuration file** for reliable walking training

## Files Created/Modified

### New Files
1. `source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/env_cfg_reliable.py`
   - Enhanced environment configuration with improved rewards, curriculum, and randomization

2. `docs/RELIABLE_WALKING_TRAINING.md`
   - Overview and conceptual guide

3. `docs/RELIABLE_WALKING_IMPLEMENTATION.md`
   - This implementation guide

### Modified Files
1. `source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/mdp/rewards.py`
   - Added new reward functions:
     - `com_stability()`: Center of mass stability reward
     - `gait_regularity()`: Regular gait pattern reward
     - `recovery_reward()`: Recovery from perturbations reward
     - `energy_efficiency()`: Energy-efficient walking reward
     - `com_height_tracking()`: Maintain target body height reward

2. `source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/mdp/curriculums.py`
   - Added curriculum functions:
     - `command_velocity_curriculum()`: Gradually increase command velocities
     - `perturbation_curriculum()`: Gradually increase perturbation magnitude

## Key Improvements

### 1. Enhanced Reward Shaping

The new reward functions encourage:
- **Stability**: CoM stability, height maintenance
- **Recovery**: Successful recovery from disturbances
- **Efficiency**: Low energy consumption
- **Regularity**: Consistent gait patterns

### 2. Curriculum Learning

Progressive difficulty increase:
- Start with conservative command velocity ranges
- Gradually increase based on success rate
- Increase perturbation magnitude as stability improves

### 3. Domain Randomization

Enhanced randomization ranges:
- Wider friction coefficient ranges (0.3-1.5)
- Larger mass variations (-1.5 to +3.0 kg)
- Broader actuator gain variations (0.7-1.3x)
- Stronger external perturbations

### 4. Observations

- Added 3-step observation history for temporal information
- Better noise models for robustness

## Usage

### Step 1: Register the Environment

You need to register the new environment configuration. Check `source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/__init__.py` and add:

```python
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg_reliable import (
    BerkeleyHumanoidLiteBipedReliableEnvCfg,
)
```

### Step 2: Create PPO Configuration

Create a new PPO config file similar to `source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/velocity/config/biped/agents/rsl_rl_ppo_cfg.py`:

```python
# rsl_rl_ppo_cfg_reliable.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class BerkeleyHumanoidLiteBipedReliablePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000  # More iterations for reliable training
    save_interval = 100
    experiment_name = "biped_reliable"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### Step 3: Register Task

In the task registration file, add:

```python
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg_reliable import (
    BerkeleyHumanoidLiteBipedReliableEnvCfg,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.agents.rsl_rl_ppo_cfg_reliable import (
    BerkeleyHumanoidLiteBipedReliablePPORunnerCfg,
)

# Register task
gym.register(
    id="Isaac-Berkeley-Humanoid-Lite-Biped-Reliable-v0",
    entry_point="berkeley_humanoid_lite.tasks.locomotion.velocity:LocomotionVelocityEnv",
    kwargs={"env_cfg_entry_point": BerkeleyHumanoidLiteBipedReliableEnvCfg},
    disable_env_checker=True,
)
```

### Step 4: Train

```bash
python scripts/rsl_rl/train.py \
    --task Isaac-Berkeley-Humanoid-Lite-Biped-Reliable-v0 \
    --num_envs 4096 \
    --max_iterations 8000
```

## Tuning Tips

### Reward Weights
Adjust reward weights in `RewardsCfg` based on training behavior:
- If robot falls frequently: Increase `recovery_reward` weight
- If gait is irregular: Increase `gait_regularity` weight
- If energy consumption is high: Increase `energy_efficiency` weight

### Curriculum Parameters
- Start with conservative command ranges
- Monitor success rate and adjust curriculum progression speed
- Increase perturbation magnitude gradually

### Domain Randomization
- Start with narrower ranges and expand as training progresses
- Monitor sim-to-real gap and adjust accordingly

## Expected Improvements

With these enhancements, you should see:
1. **More stable walking**: Better CoM control and recovery
2. **Robustness**: Better handling of perturbations and variations
3. **Efficiency**: Lower energy consumption
4. **Regularity**: More consistent gait patterns
5. **Generalization**: Better performance across different conditions

## Next Steps

1. Implement the environment registration
2. Create the PPO configuration file
3. Run initial training experiments
4. Monitor training metrics and adjust hyperparameters
5. Evaluate on hardware and iterate

## References

- IEEE Paper 11204001: Reliable Walking Training Algorithm
- Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/
- RSL-RL documentation for PPO hyperparameters

