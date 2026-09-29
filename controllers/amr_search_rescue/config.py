"""Central configuration for the autonomous mobile robot stack."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotConfig:
    wheel_radius: float = 0.033
    axle_length: float = 0.160
    robot_radius: float = 0.115
    safety_margin: float = 0.075
    max_wheel_speed: float = 9.00
    max_linear_speed: float = 0.22
    max_angular_speed: float = 1.85
    max_linear_accel: float = 0.62
    max_angular_accel: float = 3.20
    gyro_weight: float = 0.70


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
class TargetConfig:
    """Simple visual target profile supplied on the competition day.

    Webots camera images are BGRA, so channel 0 is blue, 1 is green, and 2 is
    red. Change only this profile when the organiser reveals the target.
    """

    primary_channel: int = 2
    secondary_channels: tuple[int, int] = (0, 1)
    min_primary: int = 135
    primary_ratio: float = 1.55
    primary_margin: int = 55
    min_component_area: int = 10
    seen_confidence: float = 0.28
    known_height_m: float = 0.60
    lidar_bearing_gate_deg: float = 4.5
    lidar_range_gate_min: float = 0.38
    lidar_range_gate_ratio: float = 0.38


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
class DynamicObstacleConfig:
    cluster_gap: float = 0.18
    min_cluster_points: int = 3
    min_cluster_width: float = 0.08
    max_cluster_width: float = 0.55
    association_distance: float = 0.30
    velocity_alpha: float = 0.42
    moving_speed: float = 0.12
    confirmation_hits: int = 4
    moving_confirmation_hits: int = 3
    track_timeout: float = 0.65
    obstacle_radius_min: float = 0.14
    obstacle_radius_max: float = 0.32
    prediction_horizon: float = 1.50
    planning_prediction_horizon: float = 2.20
    prediction_dt: float = 0.10
    safety_margin: float = 0.10
    uncertainty_rate: float = 0.055
    missed_uncertainty_rate: float = 0.20
    release_dwell: float = 0.50
    hazard_release_margin: float = 0.35
    lost_track_hold: float = 1.00


@dataclass(frozen=True)
class MissionConfig:
    bootstrap_rotation: float = 5.9
    bootstrap_timeout: float = 11.0
    target_stable_frames: int = 4
    required_target_count: int = 3
    target_dedup_distance: float = 1.0
    target_dedup_bearing_deg: float = 14.0
    target_track_bearing_deg: float = 45.0
    target_rearm_distance: float = 1.2
    target_stop_distance: float = 0.95
    target_confirm_time: float = 1.2
    home_tolerance: float = 0.24
    target_reacquire_timeout: float = 9.0
    mission_timeout: float = 360.0
    status_period: float = 2.0


@dataclass(frozen=True)
class SafetyConfig:
    hard_stop_distance: float = 0.18
    ttc_horizon: float = 0.85
    stuck_window: float = 3.0
    stuck_distance: float = 0.035
    recovery_duration: float = 1.8
    emergency_surface_distance: float = 0.42


@dataclass(frozen=True)
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    mapping: MapConfig = field(default_factory=MapConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    dynamic: DynamicObstacleConfig = field(default_factory=DynamicObstacleConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


CONFIG = Config()

