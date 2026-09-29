"""Wheel/gyro odometry and gated correlative scan matching."""

import math
from typing import Optional, Tuple

import numpy as np

from config import RobotConfig
from mapping import OccupancyGrid
from models import LaserScan, Pose2D, Velocity, wrap_angle


class CorrelativeScanMatcher:
    """Small-window robust scan-to-map matcher.

    This deliberately searches only around the odometry prediction. It is a
    correction stage, not a global localizer, which prevents map symmetries from
    causing large pose jumps.
    """

    def __init__(self):
        self.last_score = 0.0

    @staticmethod
    def _score(pose: Pose2D, points_robot: np.ndarray, grid: OccupancyGrid) -> float:
        if points_robot.size == 0:
            return -1.0
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        xs = pose.x + c * points_robot[:, 0] - s * points_robot[:, 1]
        ys = pose.y + s * points_robot[:, 0] + c * points_robot[:, 1]
        gx = np.rint(xs / grid.resolution).astype(np.int32) + grid.origin
        gy = np.rint(ys / grid.resolution).astype(np.int32) + grid.origin
        valid = (gx > 0) & (gx < grid.size - 1) & (gy > 0) & (gy < grid.size - 1)
        if np.count_nonzero(valid) < 12:
            return -1.0
        evidence = grid.log_odds[gy[valid], gx[valid]]
        # A trimmed positive score is robust to people and newly seen objects.
        evidence = np.sort(evidence)
        evidence = evidence[int(0.35 * evidence.size) :]
        return float(np.mean(np.tanh(evidence)))

    def match(
        self, predicted: Pose2D, scan: LaserScan, grid: OccupancyGrid
    ) -> Optional[Tuple[Pose2D, float]]:
        if grid.observed_count < 500:
            return None
        points = scan.points_robot(include_max_range=False)[::3]
        if len(points) < 20:
            return None

        baseline = self._score(predicted, points, grid)
        best_pose, best_score = predicted, baseline
        for dx in (-0.08, 0.0, 0.08):
            for dy in (-0.08, 0.0, 0.08):
                for dtheta in (-0.052, 0.0, 0.052):
                    candidate = Pose2D(
                        predicted.x + dx,
                        predicted.y + dy,
                        wrap_angle(predicted.theta + dtheta),
                    )
                    score = self._score(candidate, points, grid)
                    if score > best_score:
                        best_pose, best_score = candidate, score
        coarse = best_pose
        for dx in (-0.025, 0.0, 0.025):
            for dy in (-0.025, 0.0, 0.025):
                for dtheta in (-0.018, 0.0, 0.018):
                    candidate = Pose2D(
                        coarse.x + dx,
                        coarse.y + dy,
                        wrap_angle(coarse.theta + dtheta),
                    )
                    score = self._score(candidate, points, grid)
                    if score > best_score:
                        best_pose, best_score = candidate, score

        self.last_score = best_score
        improvement = best_score - baseline
        # Repeated corridors are highly aliased. Accept only a clear innovation
        # over the odometry prediction; a merely high absolute score is not
        # enough to move the state estimate.
        if best_score < 0.18 or improvement < 0.035:
            return None
        return best_pose, best_score


class PoseEstimator:
    def __init__(self, robot_config: RobotConfig):
        self.config = robot_config
        self.pose = Pose2D()
        self.velocity = Velocity()
        self.covariance = np.diag([1e-4, 1e-4, 2e-4]).astype(np.float64)
        self._left_previous: Optional[float] = None
        self._right_previous: Optional[float] = None
        self.matcher = CorrelativeScanMatcher()

    def update(self, left_angle: float, right_angle: float, gyro_z: float, dt: float) -> Pose2D:
        if self._left_previous is None:
            self._left_previous = left_angle
            self._right_previous = right_angle
            return self.pose
        left_distance = (left_angle - self._left_previous) * self.config.wheel_radius
        right_distance = (right_angle - self._right_previous) * self.config.wheel_radius
        self._left_previous = left_angle
        self._right_previous = right_angle

        distance = 0.5 * (left_distance + right_distance)
        wheel_rotation = (right_distance - left_distance) / self.config.axle_length
        gyro_rotation = gyro_z * dt if math.isfinite(gyro_z) and abs(gyro_z) < 6.0 else wheel_rotation
        gyro_weight = float(np.clip(self.config.gyro_weight, 0.0, 1.0))
        rotation = (1.0 - gyro_weight) * wheel_rotation + gyro_weight * gyro_rotation
        midpoint = self.pose.theta + 0.5 * rotation
        self.pose.x += distance * math.cos(midpoint)
        self.pose.y += distance * math.sin(midpoint)
        self.pose.theta = wrap_angle(self.pose.theta + rotation)
        if dt > 1e-6:
            self.velocity = Velocity(distance / dt, rotation / dt)

        heading = self.pose.theta
        jacobian = np.array(
            [[1.0, 0.0, -distance * math.sin(heading)],
             [0.0, 1.0, distance * math.cos(heading)],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        noise = np.diag(
            [2e-5 + abs(distance) * 0.002,
             2e-5 + abs(distance) * 0.002,
             2e-5 + abs(rotation) * 0.004]
        )
        self.covariance = jacobian @ self.covariance @ jacobian.T + noise
        return self.pose

    def correct_with_scan(self, scan: LaserScan, grid: OccupancyGrid) -> bool:
        match = self.matcher.match(self.pose, scan, grid)
        if match is None:
            return False
        measured, score = match
        gain = min(0.12, 0.06 + 0.08 * max(0.0, score))
        self.pose.x += gain * (measured.x - self.pose.x)
        self.pose.y += gain * (measured.y - self.pose.y)
        self.pose.theta = wrap_angle(
            self.pose.theta + gain * wrap_angle(measured.theta - self.pose.theta)
        )
        self.covariance *= 0.72
        return True

