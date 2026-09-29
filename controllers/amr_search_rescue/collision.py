"""Shared static and dynamic trajectory-clearance prediction."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from config import DynamicObstacleConfig, RobotConfig
from models import ControlCommand, DynamicObstacle, Pose2D


@dataclass(frozen=True)
class CollisionPrediction:
    min_clearance: float = math.inf
    ttc: float | None = None


def rollout_command(
    pose: Pose2D,
    command: ControlCommand,
    horizon: float,
    dt: float,
) -> np.ndarray:
    """Return rows of ``time, x, y, theta`` for a constant command."""
    steps = max(1, int(math.ceil(horizon / dt)))
    states = np.empty((steps, 4), dtype=np.float64)
    x, y, theta = pose.x, pose.y, pose.theta
    for index in range(steps):
        elapsed = min(horizon, (index + 1) * dt)
        step_dt = dt if index < steps - 1 else elapsed - index * dt
        theta += command.angular * step_dt
        x += command.linear * math.cos(theta) * step_dt
        y += command.linear * math.sin(theta) * step_dt
        states[index] = elapsed, x, y, theta
    return states


def predict_dynamic_clearance(
    pose: Pose2D,
    command: ControlCommand,
    obstacles: Sequence[DynamicObstacle],
    robot: RobotConfig,
    config: DynamicObstacleConfig,
    now: float,
    horizon: float | None = None,
    dt: float | None = None,
) -> CollisionPrediction:
    if not obstacles:
        return CollisionPrediction()
    horizon = config.prediction_horizon if horizon is None else horizon
    dt = config.prediction_dt if dt is None else dt
    states = rollout_command(pose, command, horizon, dt)
    min_clearance = math.inf
    ttc = None
    for elapsed, x, y, _ in states:
        for obstacle in obstacles:
            unseen = max(0.0, now - obstacle.last_seen)
            # Track coordinates are anchored at the last observation, not at
            # `now`.  Project through a short occlusion before rolling the
            # candidate forward or TTC will be systematically late.
            ox, oy = obstacle.predicted_position(unseen + float(elapsed))
            uncertainty = (
                (1.0 - obstacle.confidence) * 0.08
                + config.uncertainty_rate * float(elapsed)
                + config.missed_uncertainty_rate * unseen
            )
            required = robot.robot_radius + obstacle.radius + config.safety_margin + uncertainty
            clearance = math.hypot(float(x) - ox, float(y) - oy) - required
            min_clearance = min(min_clearance, clearance)
            if clearance < 0.0 and ttc is None:
                ttc = float(elapsed)
    return CollisionPrediction(min_clearance, ttc)


def minimum_static_clearance(
    pose: Pose2D,
    command: ControlCommand,
    obstacle_points: np.ndarray,
    horizon: float,
    dt: float,
) -> float:
    if obstacle_points.size == 0:
        return math.inf
    states = rollout_command(pose, command, horizon, dt)
    best = math.inf
    for _, x, y, _ in states:
        best = min(
            best,
            float(np.min(np.hypot(obstacle_points[:, 0] - x, obstacle_points[:, 1] - y))),
        )
    return best
