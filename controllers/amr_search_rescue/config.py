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
class TraversabilityConfig:
    """Parameters for bottom-up RGB floor segmentation and LiDAR fusion."""

    horizon_ratio: float = 0.42
    reference_band_ratio: float = 0.16
    chromaticity_threshold: float = 0.105
    intensity_threshold: float = 0.28
    max_vertical_gap: int = 2
    column_stride: int = 4
    lidar_bearing_gate_deg: float = 2.5
    lidar_percentile: float = 20.0


@dataclass(frozen=True)
class PlannerConfig:
    inflation_radius: float = 0.20
    frontier_min_cells: int = 7
    frontier_candidate_limit: int = 12
    frontier_reached_distance: float = 0.35
    frontier_exclusion_radius: float = 0.55
    frontier_exclusion_time: float = 20.0
    information_gain_weight: float = 0.022
    travel_cost_weight: float = 1.0
    replan_period: float = 1.0
    goal_tolerance: float = 0.20
    lookahead_distance: float = 0.62
    dwa_horizon: float = 1.35
    dwa_dt: float = 0.15
    dwa_linear_samples: int = 6
    dwa_angular_samples: int = 13
    visual_traversability_weight: float = 0.90
    return_direct_approach_distance: float = 1.80
    return_endpoint_error: float = 0.35
    return_detour_waypoint_distance: float = 0.70
    return_detour_min_travel: float = 0.30
    return_detour_duration: float = 5.0
    return_detour_reached_distance: float = 0.20
    return_detour_corridor_half_angle_deg: float = 12.0
    return_detour_max_turn_deg: float = 110.0
    breadcrumb_spacing: float = 0.20
    breadcrumb_loop_rejoin_distance: float = 0.32
    breadcrumb_loop_guard_points: int = 3
    breadcrumb_lookahead: float = 0.65
    breadcrumb_max_path_stretch: float = 2.5
    breadcrumb_path_slack: float = 0.50
    breadcrumb_reached_distance: float = 0.20
    breadcrumb_stall_progress: float = 0.08
    breadcrumb_stall_active_time: float = 4.0
    breadcrumb_revisit_radius: float = 0.42
    breadcrumb_recent_exclusion: int = 4


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
    hazard_closing_speed: float = 0.04
    hazard_receding_release_speed: float = 0.06
    evasive_safety_margin: float = 0.075
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
    pretrack_surface_distance: float = 0.55
    pretrack_closing_speed: float = 0.22
    pretrack_max_closing_speed: float = 1.20
    pretrack_min_points: int = 3
    pretrack_confirmation_frames: int = 2
    pretrack_max_angular_speed: float = 0.12
    pretrack_max_interval: float = 0.20
    pretrack_release_dwell: float = 0.18
    front_escape_angular_speed: float = 0.72
    front_escape_min_duration: float = 0.35
    front_escape_max_duration: float = 1.80
    front_escape_clearance_margin: float = 0.10
    front_escape_release_dwell: float = 0.12
    front_escape_activation_episodes: int = 3
    front_escape_activation_time: float = 5.0
    front_escape_episode_window: float = 8.0
    front_escape_anchor_radius: float = 0.20


@dataclass(frozen=True)
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    mapping: MapConfig = field(default_factory=MapConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    traversability: TraversabilityConfig = field(default_factory=TraversabilityConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    dynamic: DynamicObstacleConfig = field(default_factory=DynamicObstacleConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


CONFIG = Config()

