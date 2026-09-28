"""Independent last-line safety guard and stuck detector."""

from collections import deque
import math
from typing import Deque, Tuple

import numpy as np

from config import RobotConfig, SafetyConfig
from models import ControlCommand, LaserScan, Pose2D, angle_difference


class SafetySupervisor:
    def __init__(self, robot: RobotConfig, config: SafetyConfig):
        self.robot = robot
        self.config = config
        self.history: Deque[Tuple[float, float, float, float, float]] = deque()

    def guard(self, command: ControlCommand, scan: LaserScan) -> ControlCommand:
        finite = scan.valid_mask(include_max_range=False)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            closest_index = int(finite_indices[np.argmin(scan.ranges[finite])])
            closest_distance = float(scan.ranges[closest_index])
            closest_bearing = float(scan.angles[closest_index])
            # Reserve enough reaction distance for a laterally moving person.
            # This guard runs after DWA and therefore also covers obstacles that
            # enter the robot's side or rear during the rollout interval.
            proximity_distance = self.robot.robot_radius + 0.25
            if closest_distance < proximity_distance:
                # A fast person can enter from the side or rear after the DWA
                # rollout was scored. Move longitudinally away while turning
                # the robot's front away from the closest return.
                escape_linear = -0.15 if math.cos(closest_bearing) >= 0.0 else 0.15
                escape_turn = -1.35 if closest_bearing > 0.0 else 1.35
                return ControlCommand(escape_linear, escape_turn, "360 proximity escape")

        front = finite & (np.abs(scan.angles) < math.radians(42.0))
        if not np.any(front):
            return command
        front_distance = float(np.min(scan.ranges[front]))
        stopping_distance = (
            self.robot.robot_radius
            + self.config.hard_stop_distance
            + max(0.0, command.linear) * self.config.ttc_horizon
        )
        if front_distance >= stopping_distance:
            return command

        left = finite & (scan.angles > 0.0) & (scan.angles < math.radians(90.0))
        right = finite & (scan.angles < 0.0) & (scan.angles > -math.radians(90.0))
        left_clearance = float(np.median(scan.ranges[left])) if np.any(left) else 0.0
        right_clearance = float(np.median(scan.ranges[right])) if np.any(right) else 0.0
        turn = 0.72 if left_clearance >= right_clearance else -0.72
        reverse = -0.035 if front_distance < self.robot.robot_radius + 0.08 else 0.0
        return ControlCommand(reverse, turn, "hard safety override")

    def is_stuck(self, now: float, pose: Pose2D, command: ControlCommand) -> bool:
        self.history.append(
            (now, pose.x, pose.y, pose.theta, abs(command.linear) + 0.08 * abs(command.angular))
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

