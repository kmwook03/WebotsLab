"""Independent last-line safety guard and stuck detector."""

from collections import deque
from dataclasses import replace
import math
from typing import Deque, Tuple

import numpy as np

from collision import minimum_static_clearance, predict_dynamic_clearance, rollout_command
from config import DynamicObstacleConfig, RobotConfig, SafetyConfig
from models import (
    ControlCommand,
    DynamicObstacle,
    LaserScan,
    Pose2D,
    angle_difference,
    transform_points,
)


class SafetySupervisor:
    def __init__(
        self,
        robot: RobotConfig,
        config: SafetyConfig,
        dynamic_config: DynamicObstacleConfig,
    ):
        self.robot = robot
        self.config = config
        self.dynamic_config = dynamic_config
        self.history: Deque[Tuple[float, float, float, float, float]] = deque()
        self.override_active = False
        self.safe_since: float | None = None
        self.last_min_clearance = math.inf
        self.last_ttc: float | None = None
        self.last_override_reason = ""
        self.hazard_track_ids: set[int] = set()
        self.hazard_hold_until = -math.inf

    def _suspend_stuck_tracking(self) -> None:
        """Discard motion history while safety intentionally owns the command.

        A stationary pose is expected during a predictive stop or release
        dwell.  Retaining those samples made a correct safety yield look like
        failed actuation and immediately launched an unrelated recovery.
        """
        self.history.clear()

    def _evasive_command(
        self,
        pose: Pose2D,
        scan: LaserScan,
        obstacles: Tuple[DynamicObstacle, ...],
        now: float,
    ) -> ControlCommand:
        points_world = transform_points(scan.points_robot(include_max_range=False), pose)
        candidates = [ControlCommand(0.0, 0.0, "predictive safety stop")]
        for linear in (-0.28, -0.18, -0.10, 0.0, 0.10, 0.18, 0.28):
            for angular in (-1.35, -0.75, 0.0, 0.75, 1.35):
                if linear == 0.0 and angular == 0.0:
                    continue
                candidates.append(
                    ControlCommand(linear, angular, f"predictive evasive {linear:+.2f}/{angular:+.2f}")
                )
        recovery_config = replace(
            self.dynamic_config,
            safety_margin=0.02,
            uncertainty_rate=min(0.025, self.dynamic_config.uncertainty_rate),
            missed_uncertainty_rate=min(0.10, self.dynamic_config.missed_uncertainty_rate),
        )
        safe_candidates = []
        for candidate in candidates:
            static_clearance = minimum_static_clearance(
                pose,
                candidate,
                points_world,
                horizon=0.80,
                dt=self.dynamic_config.prediction_dt,
            )
            if static_clearance < self.robot.robot_radius + 0.02:
                continue
            end_state = rollout_command(
                pose,
                candidate,
                horizon=0.80,
                dt=self.dynamic_config.prediction_dt,
            )[-1]
            if points_world.size:
                end_static_clearance = float(
                    np.min(
                        np.hypot(
                            points_world[:, 0] - end_state[1],
                            points_world[:, 1] - end_state[2],
                        )
                    )
                )
            else:
                end_static_clearance = 5.0
            prediction = predict_dynamic_clearance(
                pose,
                candidate,
                obstacles,
                self.robot,
                recovery_config,
                now,
            )
            if prediction.ttc is not None:
                continue
            clearance = prediction.min_clearance
            if not math.isfinite(clearance):
                clearance = 5.0
            effort = abs(candidate.linear) + 0.05 * abs(candidate.angular)
            score = 2.0 * clearance + 0.8 * min(end_static_clearance, 1.0) - 0.08 * effort
            safe_candidates.append((score, clearance, end_static_clearance, candidate))
        if not safe_candidates:
            return candidates[0]
        best_score, _, _, best = max(safe_candidates, key=lambda item: item[0])
        stop_entry = next(
            (item for item in safe_candidates if item[3].reason == "predictive safety stop"),
            None,
        )
        stop_score = -math.inf if stop_entry is None else stop_entry[0]
        if best.reason != "predictive safety stop" and best_score >= stop_score + 0.03:
            return best
        return candidates[0]

    def guard(
        self,
        command: ControlCommand,
        scan: LaserScan,
        pose: Pose2D,
        dynamic_obstacles: Tuple[DynamicObstacle, ...],
        now: float,
    ) -> ControlCommand:
        finite = scan.valid_mask(include_max_range=False)
        closest_distance = float(np.min(scan.ranges[finite])) if np.any(finite) else math.inf
        front = finite & (np.abs(scan.angles) < math.radians(42.0))
        front_distance = float(np.min(scan.ranges[front])) if np.any(front) else math.inf
        stopping_distance = (
            self.robot.robot_radius
            + self.config.hard_stop_distance
            + max(0.0, command.linear) * self.config.ttc_horizon
        )
        prediction = predict_dynamic_clearance(
            pose,
            command,
            dynamic_obstacles,
            self.robot,
            self.dynamic_config,
            now,
        )
        self.last_min_clearance = prediction.min_clearance
        self.last_ttc = prediction.ttc
        compact_nearby = any(
            math.hypot(pose.x - obstacle.x, pose.y - obstacle.y)
            < self.robot.robot_radius + obstacle.radius + 0.70
            for obstacle in dynamic_obstacles
        )
        proximity_threshold = (
            self.config.emergency_surface_distance
            if compact_nearby
            else self.robot.robot_radius + 0.08
        )
        raw_emergency = closest_distance < proximity_threshold
        front_emergency = front_distance < stopping_distance
        dynamic_emergency = prediction.ttc is not None
        danger = raw_emergency or front_emergency or dynamic_emergency

        if danger:
            self.override_active = True
            self.safe_since = None
            self._suspend_stuck_tracking()
            for obstacle in dynamic_obstacles:
                distance = math.hypot(pose.x - obstacle.x, pose.y - obstacle.y)
                release_distance = (
                    self.robot.robot_radius
                    + obstacle.radius
                    + self.dynamic_config.safety_margin
                    + self.dynamic_config.hazard_release_margin
                )
                if distance < release_distance + 0.35:
                    self.hazard_track_ids.add(obstacle.track_id)
            self.hazard_hold_until = max(
                self.hazard_hold_until,
                now + self.dynamic_config.lost_track_hold,
            )
            reasons = []
            if dynamic_emergency:
                reasons.append("dynamic TTC")
            if raw_emergency:
                reasons.append("360 proximity")
            if front_emergency:
                reasons.append("front stop")
            self.last_override_reason = "+".join(reasons)
            return self._evasive_command(pose, scan, dynamic_obstacles, now)

        if self.override_active:
            self._suspend_stuck_tracking()
            tracked_hazards = tuple(
                obstacle
                for obstacle in dynamic_obstacles
                if obstacle.track_id in self.hazard_track_ids
            )
            near_hazard = any(
                math.hypot(pose.x - obstacle.x, pose.y - obstacle.y)
                < (
                    self.robot.robot_radius
                    + obstacle.radius
                    + self.dynamic_config.safety_margin
                    + self.dynamic_config.hazard_release_margin
                )
                for obstacle in tracked_hazards
            )
            if near_hazard or now < self.hazard_hold_until:
                self.safe_since = None
                self.last_override_reason = "tracked hazard yield"
                return self._evasive_command(pose, scan, tracked_hazards, now)
            if self.safe_since is None:
                self.safe_since = now
            if now - self.safe_since < self.dynamic_config.release_dwell:
                self.last_override_reason = "safety release dwell"
                return ControlCommand(0.0, 0.0, "safety release dwell")
            self.override_active = False
            self.safe_since = None
            self.hazard_track_ids.clear()
        self.last_override_reason = ""
        return command

    def is_stuck(self, now: float, pose: Pose2D, command: ControlCommand) -> bool:
        # Safety stops, evasive manoeuvres, and release dwell are deliberate
        # overrides rather than evidence that the drive failed to move.  Start
        # a fresh stuck window only after the nominal controller regains
        # ownership.
        if self.override_active or command.reason.startswith(
            ("predictive safety", "predictive evasive", "safety release", "tracked hazard")
        ):
            self._suspend_stuck_tracking()
            return False
        current_effort = abs(command.linear) + 0.08 * abs(command.angular)
        if current_effort <= 0.02:
            # Confirmation, intentional idle, and planner waits establish a
            # new motion episode.  Samples from the preceding drive must not
            # make the next departure look stuck immediately.
            self._suspend_stuck_tracking()
            return False
        self.history.append(
            (now, pose.x, pose.y, pose.theta, current_effort)
        )
        while self.history and now - self.history[0][0] > self.config.stuck_window:
            self.history.popleft()
        if len(self.history) < 4 or now - self.history[0][0] < self.config.stuck_window * 0.9:
            return False
        effort = sum(item[4] for item in self.history) / len(self.history)
        travelled = math.hypot(pose.x - self.history[0][1], pose.y - self.history[0][2])
        # Turning in place is useful progress during scan/fallback alignment;
        # include it so a deliberate rotation is not mistaken for wheel slip.
        rotated = abs(angle_difference(pose.theta, self.history[0][3]))
        motion_progress = travelled + 0.12 * rotated
        if effort > 0.08 and motion_progress < self.config.stuck_distance:
            self.history.clear()
            return True
        return False

