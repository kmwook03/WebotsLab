"""Data contracts shared by the independent robotics subsystems."""

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Optional, Sequence, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(target: float, source: float) -> float:
    return wrap_angle(target - source)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Velocity:
    linear: float = 0.0
    angular: float = 0.0


@dataclass
class ControlCommand:
    linear: float = 0.0
    angular: float = 0.0
    reason: str = "idle"


@dataclass(frozen=True)
class DynamicObstacle:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    confidence: float
    last_seen: float

    def predicted_position(self, seconds: float) -> Tuple[float, float]:
        return self.x + self.vx * seconds, self.y + self.vy * seconds


@dataclass
class DynamicObstacleFrame:
    tracks: Sequence[DynamicObstacle]
    planning_tracks: Sequence[DynamicObstacle]
    safety_tracks: Sequence[DynamicObstacle]
    ignored_hit_mask: np.ndarray
    active_track_count: int = 0

    @classmethod
    def empty(cls, scan_size: int = 0) -> "DynamicObstacleFrame":
        return cls((), (), (), np.zeros(scan_size, dtype=bool), 0)


@dataclass
class LaserScan:
    ranges: np.ndarray
    angles: np.ndarray
    min_range: float
    max_range: float

    def valid_mask(self, include_max_range: bool = True) -> np.ndarray:
        mask = np.isfinite(self.ranges) & (self.ranges >= self.min_range)
        if not include_max_range:
            mask &= self.ranges < self.max_range * 0.985
        return mask

    def points_robot(self, include_max_range: bool = False) -> np.ndarray:
        mask = self.valid_mask(include_max_range)
        ranges = np.minimum(self.ranges[mask], self.max_range)
        angles = self.angles[mask]
        if ranges.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))


@dataclass
class TargetDetection:
    seen: bool = False
    bearing: float = 0.0
    confidence: float = 0.0
    area_ratio: float = 0.0
    range_m: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class TraversabilityFrame:
    """Camera-column floor confidence with range-aligned LiDAR evidence."""

    bearings: np.ndarray
    confidence: np.ndarray
    lidar_ranges: np.ndarray
    lidar_hits: np.ndarray
    floor_mask: Optional[np.ndarray] = None

    @classmethod
    def empty(cls) -> "TraversabilityFrame":
        return cls(
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=bool),
            None,
        )


@dataclass
class PlanResult:
    path: Sequence[Tuple[float, float]]
    cost: float
    expanded: int


class MissionPhase(Enum):
    BOOTSTRAP = auto()
    EXPLORE = auto()
    TARGET_APPROACH = auto()
    CONFIRM_TARGET = auto()
    RETURN_HOME = auto()
    RECOVERY = auto()
    COMPLETE = auto()


def transform_points(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    result = np.empty_like(points, dtype=np.float32)
    result[:, 0] = pose.x + c * points[:, 0] - s * points[:, 1]
    result[:, 1] = pose.y + s * points[:, 0] + c * points[:, 1]
    return result

