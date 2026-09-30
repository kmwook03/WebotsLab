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
    Velocity,
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
        self._previous_scan_ranges: np.ndarray | None = None
        self._previous_scan_time: float | None = None
        self._pretrack_candidate_streak = 0
        self._previous_pretrack_candidates: np.ndarray | None = None
        self._active_release_dwell = dynamic_config.release_dwell
        self._front_escape_active = False
        self._front_escape_direction = 0.0
        self._front_escape_started = -math.inf
        self._front_stop_latched = False
        self._front_stop_episode_count = 0
        self._front_stop_first_episode = -math.inf
        self._front_stop_last_episode = -math.inf
        self._front_stop_anchor: tuple[float, float] | None = None

    def _register_front_stop_episode(self, pose: Pose2D, now: float) -> bool:
        """Recognise a persistent front-stop loop in one small map region."""
        if not self._front_stop_latched:
            outside_time_window = (
                now - self._front_stop_last_episode
                > self.config.front_escape_episode_window
            )
            outside_anchor = (
                self._front_stop_anchor is None
                or math.hypot(
                    pose.x - self._front_stop_anchor[0],
                    pose.y - self._front_stop_anchor[1],
                )
                > self.config.front_escape_anchor_radius
            )
            if outside_time_window or outside_anchor:
                self._front_stop_episode_count = 0
                self._front_stop_first_episode = now
                self._front_stop_anchor = (pose.x, pose.y)
            self._front_stop_episode_count += 1
            self._front_stop_last_episode = now
            self._front_stop_latched = True
        return (
            self._front_stop_episode_count
            >= self.config.front_escape_activation_episodes
            and now - self._front_stop_first_episode
            >= self.config.front_escape_activation_time
        )

    def _start_front_escape(self, scan: LaserScan, now: float) -> None:
        """Latch the turn direction with the greater near-side clearance."""
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        usable_ranges = np.where(
            np.isfinite(ranges),
            np.clip(ranges, 0.0, scan.max_range),
            scan.max_range,
        )
        side_inner = math.radians(20.0)
        side_outer = math.radians(105.0)
        left = (scan.angles >= side_inner) & (scan.angles <= side_outer)
        right = (scan.angles <= -side_inner) & (scan.angles >= -side_outer)

        def clearance_score(mask: np.ndarray) -> float:
            if not np.any(mask):
                return 0.0
            # A low percentile favours a genuinely open side rather than one
            # distant ray through a narrow gap.
            return float(np.percentile(usable_ranges[mask], 25.0))

        left_score = clearance_score(left)
        right_score = clearance_score(right)
        self._front_escape_direction = 1.0 if left_score >= right_score else -1.0
        self._front_escape_started = now
        self._front_escape_active = True

    def _front_escape_command(self) -> ControlCommand:
        return ControlCommand(
            0.0,
            self._front_escape_direction * self.config.front_escape_angular_speed,
            "front escape turn",
        )

    def _pretrack_closing_emergency(
        self,
        scan: LaserScan,
        velocity: Velocity,
        now: float,
    ) -> bool:
        """Detect a compact untracked object closing across adjacent beams.

        The expected range reduction from ego translation is removed so a
        wall approached at the commanded speed is left to the static planner.
        Rapid rotation invalidates same-beam temporal correspondence, so that
        frame only refreshes the baseline.
        """
        previous_ranges = self._previous_scan_ranges
        previous_time = self._previous_scan_time
        current_ranges = np.asarray(scan.ranges, dtype=np.float32)
        self._previous_scan_ranges = current_ranges.copy()
        self._previous_scan_time = now

        if previous_ranges is None or previous_time is None:
            self._pretrack_candidate_streak = 0
            self._previous_pretrack_candidates = None
            return False
        if previous_ranges.shape != current_ranges.shape:
            self._pretrack_candidate_streak = 0
            self._previous_pretrack_candidates = None
            return False
        elapsed = now - previous_time
        if elapsed <= 1e-6 or elapsed > self.config.pretrack_max_interval:
            self._pretrack_candidate_streak = 0
            self._previous_pretrack_candidates = None
            return False
        if abs(velocity.angular) > self.config.pretrack_max_angular_speed:
            self._pretrack_candidate_streak = 0
            self._previous_pretrack_candidates = None
            return False

        current_valid = scan.valid_mask(include_max_range=False)
        previous_valid = (
            np.isfinite(previous_ranges)
            & (previous_ranges >= scan.min_range)
            & (previous_ranges < scan.max_range * 0.985)
        )
        comparable = current_valid & previous_valid
        measured_closing = np.zeros_like(current_ranges)
        measured_closing[comparable] = (
            previous_ranges[comparable] - current_ranges[comparable]
        ) / elapsed
        ego_closing = velocity.linear * np.cos(scan.angles)
        residual_closing = measured_closing - ego_closing
        candidates = (
            comparable
            & (current_ranges < self.config.pretrack_surface_distance)
            & (residual_closing > self.config.pretrack_closing_speed)
            & (residual_closing < self.config.pretrack_max_closing_speed)
        )
        if not np.any(candidates):
            self._pretrack_candidate_streak = 0
            self._previous_pretrack_candidates = None
            return False

        run = 0
        longest_run = 0
        for candidate in candidates:
            if candidate:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 0
        # The scan wraps at +/- pi, so a rear object can occupy both ends.
        if candidates[0] and candidates[-1]:
            leading = 0
            for candidate in candidates:
                if not candidate:
                    break
                leading += 1
            trailing = 0
            for candidate in candidates[::-1]:
                if not candidate:
                    break
                trailing += 1
            longest_run = max(longest_run, leading + trailing)
        if longest_run < self.config.pretrack_min_points:
            self._pretrack_candidate_streak = 0
            self._previous_pretrack_candidates = None
            return False

        previous_candidates = self._previous_pretrack_candidates
        self._previous_pretrack_candidates = candidates.copy()
        spatially_consistent = False
        if previous_candidates is not None:
            expanded_previous = previous_candidates.copy()
            for offset in (1, 2):
                expanded_previous |= np.roll(previous_candidates, offset)
                expanded_previous |= np.roll(previous_candidates, -offset)
            spatially_consistent = bool(np.any(candidates & expanded_previous))
        self._pretrack_candidate_streak = (
            self._pretrack_candidate_streak + 1 if spatially_consistent else 1
        )
        return (
            self._pretrack_candidate_streak
            >= self.config.pretrack_confirmation_frames
        )

    @staticmethod
    def _track_kinematics(
        pose: Pose2D,
        command: ControlCommand,
        obstacle: DynamicObstacle,
        now: float,
    ) -> Tuple[float, float]:
        """Return current separation and radial closing speed for a track."""
        unseen = max(0.0, now - obstacle.last_seen)
        obstacle_x, obstacle_y = obstacle.predicted_position(unseen)
        relative_x = obstacle_x - pose.x
        relative_y = obstacle_y - pose.y
        distance = math.hypot(relative_x, relative_y)
        if distance <= 1e-6:
            return distance, math.inf
        robot_vx = command.linear * math.cos(pose.theta)
        robot_vy = command.linear * math.sin(pose.theta)
        separation_rate = (
            relative_x * (obstacle.vx - robot_vx)
            + relative_y * (obstacle.vy - robot_vy)
        ) / distance
        return distance, -separation_rate

    def _proximity_hazards(
        self,
        pose: Pose2D,
        command: ControlCommand,
        obstacles: Tuple[DynamicObstacle, ...],
        now: float,
    ) -> Tuple[DynamicObstacle, ...]:
        """Nearby tracks whose surface is closing on the robot."""
        hazards = []
        for obstacle in obstacles:
            distance, closing_speed = self._track_kinematics(
                pose, command, obstacle, now
            )
            surface_distance = distance - obstacle.radius
            if (
                surface_distance < self.config.emergency_surface_distance
                and closing_speed > self.dynamic_config.hazard_closing_speed
            ):
                hazards.append(obstacle)
        return tuple(hazards)

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
        # A moving track is evaluated by the predictive model below. Remove
        # its live LiDAR surface from the static cloud or a nearby person is
        # counted twice and every translational escape can be rejected as if
        # that person were a stationary wall.
        if points_world.size:
            static_points = np.ones(len(points_world), dtype=bool)
            for obstacle in obstacles:
                if math.hypot(obstacle.vx, obstacle.vy) < self.dynamic_config.moving_speed:
                    continue
                unseen = max(0.0, now - obstacle.last_seen)
                obstacle_x, obstacle_y = obstacle.predicted_position(unseen)
                static_points &= (
                    np.hypot(
                        points_world[:, 0] - obstacle_x,
                        points_world[:, 1] - obstacle_y,
                    )
                    > obstacle.radius + 0.10
                )
            points_world = points_world[static_points]
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
            safety_margin=self.dynamic_config.evasive_safety_margin,
        )
        safe_candidates = []
        best_effort_candidates = []
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
            clearance = prediction.min_clearance
            if not math.isfinite(clearance):
                clearance = 5.0
            final_clearance = prediction.final_clearance
            if not math.isfinite(final_clearance):
                final_clearance = 5.0
            effort = abs(candidate.linear) + 0.05 * abs(candidate.angular)
            best_effort_score = (
                3.0 * final_clearance
                + 0.6 * clearance
                + 0.8 * min(end_static_clearance, 1.0)
                - 0.08 * effort
            )
            best_effort_candidates.append(
                (best_effort_score, final_clearance, clearance, candidate)
            )
            if prediction.ttc is not None:
                continue
            score = 2.0 * clearance + 0.8 * min(end_static_clearance, 1.0) - 0.08 * effort
            safe_candidates.append((score, clearance, end_static_clearance, candidate))
        if not safe_candidates:
            if not best_effort_candidates:
                return candidates[0]
            # Once the conservative envelope is already violated, stopping is
            # not automatically safe: a person may continue walking into the
            # stationary robot. Select the statically valid command that opens
            # the greatest predicted separation by the end of the escape.
            return max(best_effort_candidates, key=lambda item: item[0])[3]
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
        current_velocity: Velocity | None = None,
    ) -> ControlCommand:
        current_velocity = current_velocity or Velocity(command.linear, command.angular)
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
        proximity_hazards = self._proximity_hazards(
            pose, command, dynamic_obstacles, now
        )
        pretrack_emergency = self._pretrack_closing_emergency(
            scan, current_velocity, now
        )
        # Keep a very small all-around hard-contact envelope for untracked
        # objects.  The wider 360-degree envelope is only justified by a
        # nearby dynamic track that is actually closing; previously any
        # compact track enlarged the threshold for every LiDAR return,
        # including side walls and people already moving away.
        contact_emergency = closest_distance < self.robot.robot_radius + 0.08
        raw_emergency = contact_emergency or bool(proximity_hazards)
        front_emergency = front_distance < stopping_distance
        dynamic_emergency = prediction.ttc is not None
        danger = raw_emergency or pretrack_emergency or front_emergency or dynamic_emergency

        if danger:
            self.override_active = True
            self.safe_since = None
            pretrack_only = pretrack_emergency and not (
                raw_emergency or front_emergency or dynamic_emergency
            )
            front_only = front_emergency and not (
                raw_emergency or pretrack_emergency or dynamic_emergency
            )
            front_escape_ready = (
                self._register_front_stop_episode(pose, now) if front_only else False
            )
            if not front_only:
                self._front_stop_latched = False
                self._front_stop_episode_count = 0
                self._front_stop_first_episode = -math.inf
                self._front_stop_anchor = None
            self._active_release_dwell = (
                self.config.pretrack_release_dwell
                if pretrack_only
                else (
                    self.config.front_escape_release_dwell
                    if front_escape_ready
                    else self.dynamic_config.release_dwell
                )
            )
            self._suspend_stuck_tracking()
            tracked_dangers = {obstacle.track_id for obstacle in proximity_hazards}
            if dynamic_emergency:
                for obstacle in dynamic_obstacles:
                    obstacle_prediction = predict_dynamic_clearance(
                        pose,
                        command,
                        (obstacle,),
                        self.robot,
                        self.dynamic_config,
                        now,
                    )
                    if obstacle_prediction.ttc is not None:
                        tracked_dangers.add(obstacle.track_id)
            if tracked_dangers:
                self.hazard_track_ids.update(tracked_dangers)
                self.hazard_hold_until = max(
                    self.hazard_hold_until,
                    now + self.dynamic_config.lost_track_hold,
                )
            reasons = []
            if dynamic_emergency:
                reasons.append("dynamic TTC")
            if raw_emergency:
                reasons.append("360 proximity")
            if pretrack_emergency:
                reasons.append("pretrack closing")
            if front_emergency:
                reasons.append("front stop")
            self.last_override_reason = "+".join(reasons)
            if pretrack_only:
                self._front_escape_active = False
                return ControlCommand(0.0, 0.0, "pretrack safety stop")
            if front_escape_ready:
                if not self._front_escape_active:
                    self._start_front_escape(scan, now)
                if now - self._front_escape_started <= self.config.front_escape_max_duration:
                    return self._front_escape_command()
            else:
                self._front_escape_active = False
            return self._evasive_command(pose, scan, dynamic_obstacles, now)

        self._front_stop_latched = False
        if self.override_active:
            self._suspend_stuck_tracking()
            tracked_hazards = tuple(
                obstacle
                for obstacle in dynamic_obstacles
                if obstacle.track_id in self.hazard_track_ids
            )
            near_hazard = False
            for obstacle in tracked_hazards:
                distance, closing_speed = self._track_kinematics(
                    pose, command, obstacle, now
                )
                if (
                    distance
                    < (
                    self.robot.robot_radius
                    + obstacle.radius
                    + self.dynamic_config.safety_margin
                    + self.dynamic_config.hazard_release_margin
                )
                    and closing_speed
                    > -self.dynamic_config.hazard_receding_release_speed
                ):
                    near_hazard = True
                    break
            visible_ids = {obstacle.track_id for obstacle in tracked_hazards}
            missing_hazard = bool(self.hazard_track_ids - visible_ids)
            if near_hazard or (missing_hazard and now < self.hazard_hold_until):
                self._front_escape_active = False
                self.safe_since = None
                self.last_override_reason = "tracked hazard yield"
                return self._evasive_command(pose, scan, tracked_hazards, now)
            if self._front_escape_active:
                escape_elapsed = now - self._front_escape_started
                release_distance = (
                    stopping_distance + self.config.front_escape_clearance_margin
                )
                if (
                    escape_elapsed < self.config.front_escape_min_duration
                    or front_distance < release_distance
                ):
                    self.last_override_reason = "front escape turn"
                    return self._front_escape_command()
                self._front_escape_active = False
                self._front_stop_episode_count = 0
                self._front_stop_first_episode = -math.inf
                self._front_stop_anchor = None
                self.safe_since = now
            if self.safe_since is None:
                self.safe_since = now
            if now - self.safe_since < self._active_release_dwell:
                self.last_override_reason = "safety release dwell"
                return ControlCommand(0.0, 0.0, "safety release dwell")
            self.override_active = False
            self.safe_since = None
            self.hazard_track_ids.clear()
            self._front_escape_active = False
        self.last_override_reason = ""
        return command

    def is_stuck(self, now: float, pose: Pose2D, command: ControlCommand) -> bool:
        # Safety stops, evasive manoeuvres, and release dwell are deliberate
        # overrides rather than evidence that the drive failed to move.  Start
        # a fresh stuck window only after the nominal controller regains
        # ownership.
        if self.override_active or command.reason.startswith(
            (
                "predictive safety",
                "predictive evasive",
                "safety release",
                "tracked hazard",
                "front escape",
            )
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

