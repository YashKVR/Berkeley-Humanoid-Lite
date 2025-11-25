"""Enhanced environment configuration for reliable walking training.

This configuration includes:
- Enhanced reward shaping for stability and recovery
- Progressive curriculum learning
- Improved domain randomization
- Better observations with history
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_BIPED_CFG, HUMANOID_LITE_LEG_JOINTS


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP with curriculum learning."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Start with conservative ranges, curriculum will expand these
            lin_vel_x=(-0.8, 0.8),  # Reduced from (-0.5, 0.5)
            lin_vel_y=(-0.15, 0.15),  # Reduced from (-0.25, 0.25)
            ang_vel_z=(-0.5, 0.5),  # Reduced from (-1.0, 1.0)
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg:
    """Enhanced observation specifications with domain randomization for IMU robustness.
    
    Domain Randomization Strategy:
    - Increased noise on IMU-related observations (base_ang_vel, projected_gravity) to simulate
      extreme real-world IMU readings (noise, drift, calibration errors)
    - Observation corruption enabled to add additional random noise during training
    - This helps the policy learn to be robust to noisy/erroneous IMU data in real-world deployment
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with enhanced features and IMU domain randomization."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.8, n_max=0.8),  # Reduced IMU-like noise for stability (was -1.5 to 1.5)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.25, n_max=0.25),  # Reduced IMU-like noise for stability (was -0.5 to 0.5)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # Enable corruption for domain randomization (IMU robustness)
            # Disable history initially to avoid numerical instability
            # Can be re-enabled once training is stable
            self.history_length = 0  # Disabled for stability

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 0  # Disabled for stability

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HUMANOID_LITE_LEG_JOINTS,
        scale=0.25,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Enhanced reward terms for reliable walking."""

    # === Task-space performance ===
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=15.0,  # Very strongly encourage following velocity commands
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=1.0,
    )

    # === Stability and reliability rewards ===
    com_stability = RewTerm(
        func=mdp.com_stability,
        weight=1.0,  # Encourage stable CoM
    )
    recovery_reward = RewTerm(
        func=mdp.recovery_reward,
        weight=2.0,  # Strong reward for recovery
    )
    com_height_tracking = RewTerm(
        func=mdp.com_height_tracking,
        params={"target_height": 0.85},
        weight=1.0,
    )
    
    # === Penguin-like gait rewards ===
    # Strongly minimize hip roll and yaw movement (keep legs separated, no rotation)
    minimal_hip_roll_yaw = RewTerm(
        func=mdp.leg_separation_reward,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]),
        },
        weight=4.0,  # Strong weight to minimize hip roll/yaw movement
    )
    # Prevent knee straightening (maintain bent knee posture)
    prevent_knee_straight = RewTerm(
        func=mdp.prevent_knee_straightening,
        params={
            "min_knee_angle": 0.7,  # Slightly reduced minimum (was 0.8) to allow more flexibility
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"]),
        },
        weight=-2.0,  # Reduced penalty for stability (was -3.0)
    )
    # Allow controlled movement in locomotion joints (hip pitch, knee, ankle pitch)
    locomotion_joints = RewTerm(
        func=mdp.allow_locomotion_joint_movement,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_pitch_joint", ".*_ankle_pitch_joint"]),
            "max_deviation": 0.3,  # Allow up to 0.3 rad deviation for locomotion
        },
        weight=1.0,  # Encourage staying close but allow necessary movement
    )
    # Keep both feet in contact (prevent single-leg stance) - but allow brief stepping
    dual_stance = RewTerm(
        func=mdp.dual_stance_reward,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
        },
        weight=0.5,  # Allow stepping for forward motion
    )

    # === Basic behaviors ===
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.1,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,  # Reduced penalty for stability (was -2.0)
    )

    # === Motion quality ===
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-2.0e-3,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-1.0e-6,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )
    
    # === Energy efficiency ===
    energy_efficiency = RewTerm(
        func=mdp.energy_efficiency,
        weight=0.5,  # Encourage efficient walking
    )

    # === Gait quality ===
    # Allow stepping for forward motion (penguin-like gait with minimal stepping)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.2,  # Allow stepping for forward motion
        },
        weight=1.5,  # Increased to strongly encourage stepping for forward motion
    )
    gait_regularity = RewTerm(
        func=mdp.gait_regularity,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.3,
        },
        weight=1.0,
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
        weight=-0.1,
    )

    # === Safety ===
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        weight=-0.2,
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
        weight=-0.2,
    )
    
    # Penalize deviation from default knee position (encourage maintaining bent knee posture)
    straight_knee_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_pitch_joint"]),
        },
        weight=-0.5,  # Penalize deviation from default bent knee position
    )


@configclass
class TerminationsCfg:
    """Termination conditions with safety margins."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg("robot", body_names="base")},  # More lenient (was 0.78)
    )


@configclass
class EventsCfg:
    """Enhanced domain randomization for robustness."""

    # === Startup behaviors ===
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),  # Reduced range for stability (was 0.3 to 1.5)
            "dynamic_friction_range": (0.5, 1.2),  # Reduced range for stability (was 0.3 to 1.5)
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        mode="startup",
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 1.0),  # Reduced range for stability (was -1.5 to 3.0)
            "operation": "add",
        },
        mode="startup",
    )
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.08, 0.08),  # Wider range
            "operation": "add",
        },
        mode="startup",
    )
    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_torque_constant,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "torque_constant_params": (0.7, 1.3),  # Wider range
            "operation": "scale",
        },
        mode="startup",
    )

    # === Reset behaviors ===
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),  # Tighter range to maintain penguin-like posture
            "velocity_range": (0.0, 0.0),
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-1.5, 1.5),  # Reduced perturbations for stability (was -3.0 to 3.0)
            "torque_range": (-1.5, 1.5),  # Reduced perturbations for stability (was -3.0 to 3.0)
        },
        mode="reset",
    )

    # === Interval behaviors (periodic perturbations) ===
    # Note: push_by_setting_velocity may need to be implemented in events.py
    # For now, using external force/torque at reset which provides similar effect


@configclass
class CurriculumsCfg:
    """Curriculum learning for progressive difficulty."""

    command_velocity = CurriculumTerm(
        func=mdp.command_velocity_curriculum,
        params={"command_name": "base_velocity"},
    )
    perturbation = CurriculumTerm(
        func=mdp.perturbation_curriculum,
    )


@configclass
class BerkeleyHumanoidLiteBipedReliableEnvCfg(LocomotionVelocityEnvCfg):
    """Enhanced environment configuration for reliable walking training."""

    # Policy commands
    commands: CommandsCfg = CommandsCfg()

    # Policy observations
    observations: ObservationsCfg = ObservationsCfg()

    # Policy actions
    actions: ActionsCfg = ActionsCfg()

    # Policy rewards
    rewards: RewardsCfg = RewardsCfg()

    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg()

    # Randomization events
    events: EventsCfg = EventsCfg()

    # Curriculums
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Physics settings
        # 25 Hz override
        self.decimation = 8

        # Scene
        self.scene.robot = HUMANOID_LITE_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        
        # Use standard number of environments for stability
        # Can increase once training is stable
        self.scene.num_envs = 2048  # Reduced from 4096 for initial stability
        self.max_episode_length_s = 20.0  # Longer episodes

