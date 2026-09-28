"""Central configuration for the autonomous mobile robot stack."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotConfig:
    wheel_radius: float = 0.033
    axle_length: float = 0.160
    robot_radius: float = 0.115
    safety_margin: float = 0.075
    max_wheel_speed: float = 9.00
    max_linear_speed: float = 0.285
    max_angular_speed: float = 1.85
    max_linear_accel: float = 0.62
    max_angular_accel: float = 3.20


@dataclass(frozen=True)
class MapConfig:
    size: int = 280
    resolution: float = 0.05
    free_log_odds: float = -0.38
    occupied_log_odds: float = 0.82
    min_log_odds: float = -4.0
    max_log_odds: float = 4.0
    occupied_threshold: float = 0.62
    free_threshold: float = -0.45
    ray_stride: int = 3
    stale_hit_steps: int = 240
    stale_decay: float = 0.96


@dataclass(frozen=True)
class PlannerConfig:
    inflation_radius: float = 0.20
    frontier_min_cells: int = 7
    frontier_candidate_limit: int = 12
    information_gain_weight: float = 0.022
    travel_cost_weight: float = 1.0
    replan_period: float = 1.0
    goal_tolerance: float = 0.20
    lookahead_distance: float = 0.62
    dwa_horizon: float = 1.35
    dwa_dt: float = 0.15
    dwa_linear_samples: int = 6
    dwa_angular_samples: int = 13


@dataclass(frozen=True)
class MissionConfig:
    bootstrap_rotation: float = 5.9
    bootstrap_timeout: float = 11.0
    target_stable_frames: int = 4
    target_stop_distance: float = 0.95
    target_confirm_time: float = 1.2
    home_tolerance: float = 0.24
    target_reacquire_timeout: float = 9.0
    mission_timeout: float = 280.0
    status_period: float = 2.0


@dataclass(frozen=True)
class SafetyConfig:
    hard_stop_distance: float = 0.18
    ttc_horizon: float = 0.85
    stuck_window: float = 3.0
    stuck_distance: float = 0.035
    recovery_duration: float = 1.8


@dataclass(frozen=True)
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    mapping: MapConfig = field(default_factory=MapConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


CONFIG = Config()

