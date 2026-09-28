"""Explicit mission state machine for search, rescue, and return."""

import math
from typing import Optional, Tuple

from config import MissionConfig
from models import MissionPhase, Pose2D, TargetDetection, angle_difference


WorldPoint = Tuple[float, float]


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

    def update(self, now: float, pose: Pose2D, detection: TargetDetection) -> Optional[str]:
        self.transition_message = ""
        delta_theta = abs(angle_difference(pose.theta, self._last_theta))
        self._last_theta = pose.theta

        accept_detection = False
        object_observation: Optional[WorldPoint] = None
        if detection.seen and detection.range_m is not None:
            angle = pose.theta + detection.bearing
            object_observation = (
                pose.x + detection.range_m * math.cos(angle),
                pose.y + detection.range_m * math.sin(angle),
            )
            already_visited = any(
                math.hypot(object_observation[0] - target[0], object_observation[1] - target[1])
                <= self.config.target_dedup_distance
                for target in self.visited_targets
            )
            rearmed = self.last_confirmed_pose is None or math.hypot(
                pose.x - self.last_confirmed_pose.x,
                pose.y - self.last_confirmed_pose.y,
            ) >= self.config.target_rearm_distance
            accept_detection = self.phase in (
                MissionPhase.TARGET_APPROACH,
                MissionPhase.CONFIRM_TARGET,
            ) or (not already_visited and rearmed)

        if accept_detection and object_observation is not None:
            self._target_frames += 1
            self.last_target_time = now
            angle = pose.theta + detection.bearing
            stand_off = max(0.0, detection.range_m - self.config.target_stop_distance)
            observation = (
                pose.x + stand_off * math.cos(angle),
                pose.y + stand_off * math.sin(angle),
            )
            if self.target_estimate is None:
                self.target_estimate = observation
                self.target_object_estimate = object_observation
            else:
                self.target_estimate = (
                    0.68 * self.target_estimate[0] + 0.32 * observation[0],
                    0.68 * self.target_estimate[1] + 0.32 * observation[1],
                )
                assert self.target_object_estimate is not None
                self.target_object_estimate = (
                    0.68 * self.target_object_estimate[0] + 0.32 * object_observation[0],
                    0.68 * self.target_object_estimate[1] + 0.32 * object_observation[1],
                )
        else:
            self._target_frames = max(0, self._target_frames - 1)

        if self.phase == MissionPhase.BOOTSTRAP:
            self._rotation_accumulated += delta_theta
            if (
                self._rotation_accumulated >= self.config.bootstrap_rotation
                or now - self.phase_started_at >= self.config.bootstrap_timeout
            ):
                self._transition(MissionPhase.EXPLORE, now, "initial 360-degree scan complete")

        if self.phase == MissionPhase.EXPLORE and self._target_frames >= self.config.target_stable_frames:
            self._transition(MissionPhase.TARGET_APPROACH, now, "visual target confirmed")

        if self.phase == MissionPhase.TARGET_APPROACH:
            if detection.seen and detection.range_m is not None and detection.range_m <= self.config.target_stop_distance + 0.05:
                self._transition(MissionPhase.CONFIRM_TARGET, now, "target stand-off reached")
            elif (
                self.target_estimate is not None
                and now - self.last_target_time > self.config.target_reacquire_timeout
            ):
                self.target_estimate = None
                self.target_object_estimate = None
                self._target_frames = 0
                self._transition(MissionPhase.EXPLORE, now, "target observation lost; resume search")

        if self.phase == MissionPhase.CONFIRM_TARGET:
            if now - self.phase_started_at >= self.config.target_confirm_time:
                if self.target_object_estimate is not None and not any(
                    math.hypot(
                        self.target_object_estimate[0] - target[0],
                        self.target_object_estimate[1] - target[1],
                    )
                    <= self.config.target_dedup_distance
                    for target in self.visited_targets
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
                    self.target_estimate = None
                    self.target_object_estimate = None
                    self._target_frames = 0
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

