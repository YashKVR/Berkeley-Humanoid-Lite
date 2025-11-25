"""PPO configuration for reliable walking training.

This configuration is optimized for training a robust and reliable walking policy
with enhanced stability, recovery, and energy efficiency.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BerkeleyHumanoidLiteBipedReliablePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for reliable walking training.
    
    This configuration uses:
    - More training iterations for better convergence
    - Same network architecture as base config
    - Optimized hyperparameters for stability and recovery learning
    """
    num_steps_per_env = 24
    max_iterations = 8000  # More iterations for reliable training
    save_interval = 100
    experiment_name = "biped_reliable"
    empirical_normalization = True  # Enable to stabilize training
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Reduced from 1.0 for more stable initial exploration
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
        learning_rate=5.0e-4,  # Reduced from 1.0e-3 for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,  # Reduced from 1.0 for more aggressive gradient clipping
    )

