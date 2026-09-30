"""Explicit mission state machine for search, rescue, and return."""

import math
from typing import Optional, Tuple

from config import MissionConfig
from models import MissionPhase, Pose2D, TargetDetection, angle_difference


WorldPoint = Tuple[float, float]
TARGET_TRACKING_PHASES = (MissionPhase.TARGET_APPROACH, MissionPhase.CONFIRM_TARGET)


class MissionManager:
    def __init__(self, config: MissionConfig):
        self.config = config
        self.phase = MissionPhase.BOOTSTRAP
        self.previous_phase = self.phase
        self.start_pose = Pose2D()
        self.started_at = 0.0
        self.phase_started_at = 0.0
        self._last_theta = 0.0
        self._rotation_accumulated = 0.0
        self._target_frames = 0
        self.target_estimate: Optional[WorldPoint] = None
        self.target_object_estimate: Optional[WorldPoint] = None
        self.visited_targets: list[WorldPoint] = []
        self.last_confirmed_pose: Optional[Pose2D] = None
        self.last_target_time = -math.inf
        self.target_observation_accepted = False
        self.transition_message = ""

    @property
    def visited_count(self) -> int:
        return len(self.visited_targets)

    def start(self, now: float, pose: Pose2D) -> None:
        self.started_at = now
        self.phase_started_at = now
        self.start_pose = Pose2D(pose.x, pose.y, pose.theta)
        self._last_theta = pose.theta

    def _transition(self, phase: MissionPhase, now: float, reason: str) -> None:
        if self.phase == phase:
            return
        old = self.phase
        self.previous_phase = old
        self.phase = phase
        self.phase_started_at = now
        self.transition_message = f"{old.name} -> {phase.name}: {reason}"

    def _matches_visited_target(
        self,
        pose: Pose2D,
        detection: TargetDetection,
        object_observation: Optional[WorldPoint],
    ) -> bool:
        if not self.visited_targets:
            return False
        if object_observation is not None and self._is_near_visited(object_observation):
            return True

        # Bearing is meaningful only while the camera is actively observing a
        # target.  During a one-frame dropout TargetDetection carries its
        # default zero bearing, which must not reject a valid in-progress
        # approach by accident.
        if not detection.seen:
            return False

        bearing_gate = math.radians(self.config.target_dedup_bearing_deg)
        for target in self.visited_targets:
            predicted_bearing = angle_difference(
                math.atan2(target[1] - pose.y, target[0] - pose.x),
                pose.theta,
            )
            if abs(angle_difference(detection.bearing, predicted_bearing)) <= bearing_gate:
                return True
        return False

    def _is_near_visited(self, point: WorldPoint) -> bool:
        return any(
            math.hypot(point[0] - target[0], point[1] - target[1])
            <= self.config.target_dedup_distance
            for target in self.visited_targets
        )

    @staticmethod
    def _project(pose: Pose2D, bearing: float, distance: float) -> WorldPoint:
        angle = pose.theta + bearing
        return (
            pose.x + distance * math.cos(angle),
            pose.y + distance * math.sin(angle),
        )

    @staticmethod
    def _smooth(previous: WorldPoint, observed: WorldPoint) -> WorldPoint:
        return (
            0.68 * previous[0] + 0.32 * observed[0],
            0.68 * previous[1] + 0.32 * observed[1],
        )

    def _clear_target_candidate(self) -> None:
        self.target_estimate = None
        self.target_object_estimate = None
        self._target_frames = 0

    def _evaluate_detection(
        self,
        pose: Pose2D,
        detection: TargetDetection,
    ) -> Tuple[bool, Optional[WorldPoint]]:
        """Associate a visible target with the current candidate or reject it."""
        if not detection.seen or detection.range_m is None:
            return False, None

        observation = self._project(pose, detection.bearing, detection.range_m)
        already_visited = self._matches_visited_target(pose, detection, observation)
        rearmed = self.last_confirmed_pose is None or math.hypot(
            pose.x - self.last_confirmed_pose.x,
            pose.y - self.last_confirmed_pose.y,
        ) >= self.config.target_rearm_distance

        associated = True
        if (
            self.phase in TARGET_TRACKING_PHASES
            and self.target_object_estimate is not None
        ):
            expected_bearing = angle_difference(
                math.atan2(
                    self.target_object_estimate[1] - pose.y,
                    self.target_object_estimate[0] - pose.x,
                ),
                pose.theta,
            )
            associated = abs(
                angle_difference(detection.bearing, expected_bearing)
            ) <= math.radians(self.config.target_track_bearing_deg)

        if self.phase in TARGET_TRACKING_PHASES:
            return associated and not already_visited, observation
        return not already_visited and rearmed, observation

    def _update_target_candidate(
        self,
        now: float,
        pose: Pose2D,
        detection: TargetDetection,
        accepted: bool,
        object_observation: Optional[WorldPoint],
    ) -> None:
        if not accepted or object_observation is None:
            self._target_frames = max(0, self._target_frames - 1)
            return

        assert detection.range_m is not None
        self._target_frames += 1
        self.last_target_time = now
        stand_off = max(0.0, detection.range_m - self.config.target_stop_distance)
        approach_observation = self._project(pose, detection.bearing, stand_off)
        if self.target_estimate is None:
            self.target_estimate = approach_observation
            self.target_object_estimate = object_observation
            return

        self.target_estimate = self._smooth(
            self.target_estimate, approach_observation
        )
        assert self.target_object_estimate is not None
        self.target_object_estimate = self._smooth(
            self.target_object_estimate, object_observation
        )

    def update(self, now: float, pose: Pose2D, detection: TargetDetection) -> Optional[str]:
        self.transition_message = ""
        self.target_observation_accepted = False
        delta_theta = abs(angle_difference(pose.theta, self._last_theta))
        self._last_theta = pose.theta

        accept_detection, object_observation = self._evaluate_detection(pose, detection)
        self.target_observation_accepted = accept_detection
        self._update_target_candidate(
            now, pose, detection, accept_detection, object_observation
        )

        if self.phase == MissionPhase.BOOTSTRAP:
            self._rotation_accumulated += delta_theta
            if (
                self._rotation_accumulated >= self.config.bootstrap_rotation
                or now - self.phase_started_at >= self.config.bootstrap_timeout
            ):
                self._transition(MissionPhase.EXPLORE, now, "initial 360-degree scan complete")

        if self.phase == MissionPhase.EXPLORE and self._target_frames >= self.config.target_stable_frames:
            if self._matches_visited_target(pose, detection, self.target_object_estimate):
                self._clear_target_candidate()
            else:
                self._transition(MissionPhase.TARGET_APPROACH, now, "visual target confirmed")

        if self.phase == MissionPhase.TARGET_APPROACH:
            if (
                accept_detection
                and detection.range_m is not None
                and detection.range_m <= self.config.target_stop_distance + 0.05
            ):
                self._transition(MissionPhase.CONFIRM_TARGET, now, "target stand-off reached")
            elif (
                self.target_estimate is not None
                and now - self.last_target_time > self.config.target_reacquire_timeout
            ):
                self._clear_target_candidate()
                self._transition(MissionPhase.EXPLORE, now, "target observation lost; resume search")

        if self.phase == MissionPhase.CONFIRM_TARGET:
            if now - self.phase_started_at >= self.config.target_confirm_time:
                if (
                    self.target_object_estimate is not None
                    and not self._is_near_visited(self.target_object_estimate)
                ):
                    self.visited_targets.append(self.target_object_estimate)
                    self.last_confirmed_pose = Pose2D(pose.x, pose.y, pose.theta)

                if self.visited_count >= max(1, self.config.required_target_count):
                    self._transition(
                        MissionPhase.RETURN_HOME,
                        now,
                        f"all {self.visited_count} rescue targets confirmed",
                    )
                else:
                    confirmed = self.visited_count
                    self._clear_target_candidate()
                    self._transition(
                        MissionPhase.EXPLORE,
                        now,
                        f"target {confirmed}/{self.config.required_target_count} confirmed; resume search",
                    )

        if self.phase == MissionPhase.RETURN_HOME:
            if math.hypot(pose.x - self.start_pose.x, pose.y - self.start_pose.y) <= self.config.home_tolerance:
                self._transition(MissionPhase.COMPLETE, now, "start pose reached")

        if (
            now - self.started_at > self.config.mission_timeout
            and self.phase not in (MissionPhase.RETURN_HOME, MissionPhase.COMPLETE)
        ):
            self._transition(MissionPhase.RETURN_HOME, now, "mission timeout; fail-safe return")
        return self.transition_message or None

    def enter_recovery(self, now: float, reason: str) -> None:
        if self.phase in (MissionPhase.COMPLETE, MissionPhase.CONFIRM_TARGET, MissionPhase.RECOVERY):
            return
        self.previous_phase = self.phase
        self._transition(MissionPhase.RECOVERY, now, reason)

    def finish_recovery(self, now: float) -> None:
        if self.phase == MissionPhase.RECOVERY:
            resume = self.previous_phase
            self._transition(resume, now, "recovery manoeuvre complete")

    def goal(self, frontier_goal: Optional[WorldPoint]) -> Optional[WorldPoint]:
        if self.phase == MissionPhase.EXPLORE:
            return frontier_goal
        if self.phase == MissionPhase.TARGET_APPROACH:
            return self.target_estimate
        if self.phase == MissionPhase.RETURN_HOME:
            return self.start_pose.x, self.start_pose.y
        return None

