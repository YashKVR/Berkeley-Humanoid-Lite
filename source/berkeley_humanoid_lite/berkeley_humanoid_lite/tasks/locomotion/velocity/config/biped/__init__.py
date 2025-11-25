import gymnasium as gym

from . import env_cfg, agents
from . import env_cfg_reliable

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Berkeley-Humanoid-Lite-Biped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.BerkeleyHumanoidLiteBipedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BerkeleyHumanoidLiteBipedPPORunnerCfg,
    },
)

# Register reliable walking variant
gym.register(
    id="Velocity-Berkeley-Humanoid-Lite-Biped-Reliable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_reliable.BerkeleyHumanoidLiteBipedReliableEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg_reliable.BerkeleyHumanoidLiteBipedReliablePPORunnerCfg,
    },
)
