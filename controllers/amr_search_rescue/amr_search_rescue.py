"""Webots adapter for the modular autonomous search-and-rescue stack."""

import json
import math
import traceback
from typing import List, Optional, Tuple

import numpy as np
from controller import Robot

from config import CONFIG
from localization import PoseEstimator
from mapping import OccupancyGrid
from mission import MissionManager
from models import (
    ControlCommand,
    DynamicObstacleFrame,
    LaserScan,
    MissionPhase,
    TargetDetection,
    angle_difference,
    wrap_angle,
)
from perception import RedTargetDetector
from planning import AStarPlanner, DynamicWindowPlanner, FrontierExplorer
from safety import SafetySupervisor
from tracking import DynamicObstacleTracker
from visualization import MapDisplay


class AutonomousSearchAndRescue:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_sensor = self.robot.getDevice("left wheel sensor")
        self.right_sensor = self.robot.getDevice("right wheel sensor")
        self.left_sensor.enable(self.time_step)
        self.right_sensor.enable(self.time_step)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()
        resolution = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        # Webots range images are ordered from +FOV/2 to -FOV/2.
        self.lidar_angles = np.linspace(0.5 * fov, -0.5 * fov, resolution, dtype=np.float32)
        self.lidar_min_range = self.lidar.getMinRange()
        self.lidar_max_range = self.lidar.getMaxRange()
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)
        self.display = MapDisplay(self.robot.getDevice("map display"))
        self.emitter = self.robot.getDevice("mission emitter")

        self.grid = OccupancyGrid(CONFIG.mapping)
        self.estimator = PoseEstimator(CONFIG.robot)
        self.target_detector = RedTargetDetector(CONFIG.target)
        self.global_planner = AStarPlanner(CONFIG.planner)
        self.explorer = FrontierExplorer(CONFIG.planner, self.global_planner)
        self.local_planner = DynamicWindowPlanner(CONFIG.robot, CONFIG.planner, CONFIG.dynamic)
        self.dynamic_tracker = DynamicObstacleTracker(CONFIG.dynamic)
        self.dynamic_frame = DynamicObstacleFrame.empty(resolution)
        self.safety = SafetySupervisor(CONFIG.robot, CONFIG.safety, CONFIG.dynamic)
        self.mission = MissionManager(CONFIG.mission)

        self.path: List[Tuple[float, float]] = []
        self.frontier_goal: Optional[Tuple[float, float]] = None
        self.current_goal: Optional[Tuple[float, float]] = None
        self.last_plan_time = -math.inf
        self.last_status_time = -math.inf
        self.last_detection = TargetDetection()
        self.last_command = ControlCommand()
        self.fallback_heading: Optional[float] = None
        self.fallback_until = -math.inf
        self.departure_until = -math.inf
        self.last_visited_count = 0
        self.step_count = 0

    def _scan(self) -> LaserScan:
        ranges = np.asarray(self.lidar.getRangeImage(), dtype=np.float32)
        return LaserScan(ranges, self.lidar_angles, self.lidar_min_range, self.lidar_max_range)

    def _detection(self, scan: LaserScan) -> TargetDetection:
        return self.target_detector.detect(
            self.camera.getImage(),
            self.camera.getWidth(),
            self.camera.getHeight(),
            self.camera.getFov(),
            scan,
        )

    def _path_invalid(self) -> bool:
        if not self.path:
            return True
        blocked = self.grid.inflated_obstacles(CONFIG.planner.inflation_radius)
        for point in self.path[1:]:
            x, y = self.grid.world_to_grid(*point)
            if not self.grid.in_bounds((x, y)) or blocked[y, x]:
                return True
        return False

    def _prune_path(self) -> None:
        pose = self.estimator.pose
        while len(self.path) > 1 and math.hypot(self.path[0][0] - pose.x, self.path[0][1] - pose.y) < 0.22:
            self.path.pop(0)

    @staticmethod
    def _goal_changed(first, second, threshold: float = 0.28) -> bool:
        if first is None or second is None:
            return first != second
        return math.hypot(first[0] - second[0], first[1] - second[1]) > threshold

    def _update_plan(self, now: float) -> None:
        pose = self.estimator.pose
        phase = self.mission.phase
        periodic = now - self.last_plan_time >= CONFIG.planner.replan_period

        if phase == MissionPhase.EXPLORE:
            reached = self.frontier_goal is not None and math.hypot(
                pose.x - self.frontier_goal[0], pose.y - self.frontier_goal[1]
            ) < CONFIG.planner.goal_tolerance
            if reached or self.frontier_goal is None or (periodic and self._path_invalid()):
                self.frontier_goal, result = self.explorer.choose(self.grid, pose)
                self.path = list(result.path) if result is not None else []
                self.current_goal = self.frontier_goal
                self.last_plan_time = now
            return

        desired = self.mission.goal(self.frontier_goal)
        if desired is None:
            self.path = []
            self.current_goal = None
            return
        endpoint_reached = self.path and math.hypot(
            pose.x - self.path[-1][0], pose.y - self.path[-1][1]
        ) < CONFIG.planner.goal_tolerance
        if self._goal_changed(desired, self.current_goal) or endpoint_reached or (periodic and self._path_invalid()):
            result = self.global_planner.plan(self.grid, (pose.x, pose.y), desired)
            self.path = list(result.path) if result is not None else []
            self.current_goal = desired
            self.last_plan_time = now

    def _nominal_command(self, now: float, scan: LaserScan) -> ControlCommand:
        phase = self.mission.phase
        if phase == MissionPhase.BOOTSTRAP:
            return ControlCommand(0.0, 0.72, "bootstrap scan")
        if phase == MissionPhase.CONFIRM_TARGET:
            return ControlCommand(0.0, 0.0, "confirm target")
        if phase == MissionPhase.COMPLETE:
            return ControlCommand(0.0, 0.0, "mission complete")
        if phase == MissionPhase.RECOVERY:
            elapsed = now - self.mission.phase_started_at
            if elapsed >= CONFIG.safety.recovery_duration:
                self.mission.finish_recovery(now)
                self.path = []
                self.current_goal = None
                return ControlCommand()
            turn = 0.95 if int(self.step_count / 12) % 2 == 0 else -0.95
            return ControlCommand(-0.055, turn, "stuck recovery")

        # A confirmed object remains in the camera while the robot is at its
        # stand-off point. Back out along the already observed approach lane so
        # the detector can re-arm for the next object instead of orbiting the
        # same one. The independent 360-degree guard still owns final safety.
        if phase == MissionPhase.EXPLORE and now < self.departure_until:
            return ControlCommand(-0.18, 0.16, "leave confirmed target")

        # Once the mission manager has associated the current camera blob with
        # the locked, unvisited target, use its bearing directly.  Monocular
        # range can be deliberately conservative at long distance; letting an
        # A* path to that noisy endpoint take precedence made the robot orbit
        # an already mapped area instead of closing on the visible marker.
        if (
            phase == MissionPhase.TARGET_APPROACH
            and self.last_detection.seen
            and self.mission.target_observation_accepted
        ):
            angular = float(np.clip(1.7 * self.last_detection.bearing, -0.8, 0.8))
            range_m = self.last_detection.range_m or CONFIG.mission.target_stop_distance
            range_error = max(0.0, range_m - CONFIG.mission.target_stop_distance)
            approach_speed = float(np.clip(0.09 + 0.045 * range_error, 0.09, 0.22))
            heading_scale = float(np.clip(1.0 - abs(self.last_detection.bearing) / 0.75, 0.35, 1.0))
            visual_command = ControlCommand(
                approach_speed * heading_scale,
                angular,
                "visual target servo",
            )
            if not self.dynamic_frame.planning_tracks:
                return visual_command

            # A provisional moving track should influence target approach
            # before it escalates into a safety override.  Build a short local
            # path from the live camera bearing so DWA can slow or steer around
            # the mover without trusting the noisy long-range monocular goal.
            pose = self.estimator.pose
            local_distance = float(np.clip(range_m, 0.8, 2.0))
            local_heading = pose.theta + self.last_detection.bearing
            visual_goal = (
                pose.x + local_distance * math.cos(local_heading),
                pose.y + local_distance * math.sin(local_heading),
            )
            planned = self.local_planner.command(
                pose,
                self.estimator.velocity,
                [(pose.x, pose.y), visual_goal],
                scan,
                self.dt,
                self.dynamic_frame.planning_tracks,
                now,
            )
            return ControlCommand(planned.linear, planned.angular, "visual DWA")

        self._update_plan(now)
        self._prune_path()
        if self.path:
            self.fallback_heading = None
            return self.local_planner.command(
                self.estimator.pose,
                self.estimator.velocity,
                self.path,
                scan,
                self.dt,
                self.dynamic_frame.planning_tracks,
                now,
            )
        if phase == MissionPhase.EXPLORE:
            if self.fallback_heading is None or now >= self.fallback_until:
                clearance = np.asarray(scan.ranges, dtype=np.float32).copy()
                clearance[~np.isfinite(clearance)] = scan.max_range
                np.clip(clearance, 0.0, scan.max_range, out=clearance)
                kernel = np.ones(21, dtype=np.float32) / 21.0
                smoothed = np.convolve(clearance, kernel, mode="same")
                best_index = int(np.argmax(smoothed))
                bearing = float(scan.angles[best_index])
                self.fallback_heading = wrap_angle(self.estimator.pose.theta + bearing)
                self.fallback_until = now + 8.0
            heading_error = angle_difference(self.fallback_heading, self.estimator.pose.theta)
            angular = float(np.clip(1.35 * heading_error, -0.90, 0.90))
            aligned = max(0.0, 1.0 - abs(heading_error) / 0.65)
            linear = 0.0 if abs(heading_error) > 0.50 else min(
                CONFIG.robot.max_linear_speed,
                0.10 + 0.10 * aligned,
            )
            return ControlCommand(linear, angular, "open-space exploration fallback")
        return ControlCommand(0.0, 0.52, "search for reachable frontier")

    def _actuate(self, command: ControlCommand) -> None:
        half_track = 0.5 * CONFIG.robot.axle_length
        left = (command.linear - command.angular * half_track) / CONFIG.robot.wheel_radius
        right = (command.linear + command.angular * half_track) / CONFIG.robot.wheel_radius
        scale = max(1.0, abs(left) / CONFIG.robot.max_wheel_speed, abs(right) / CONFIG.robot.max_wheel_speed)
        self.left_motor.setVelocity(left / scale)
        self.right_motor.setVelocity(right / scale)

    def _publish(self, now: float) -> None:
        if now - self.last_status_time < CONFIG.mission.status_period:
            return
        self.last_status_time = now
        pose = self.estimator.pose
        status = {
            "time": round(now, 2),
            "phase": self.mission.phase.name,
            "pose": [round(pose.x, 3), round(pose.y, 3), round(pose.theta, 3)],
            "map_cells": self.grid.observed_count,
            "target_confidence": round(self.last_detection.confidence, 3),
            "target_range": None if self.last_detection.range_m is None else round(self.last_detection.range_m, 3),
            "targets_visited": self.mission.visited_count,
            "targets_required": CONFIG.mission.required_target_count,
            "command": [round(self.last_command.linear, 3), round(self.last_command.angular, 3)],
            "reason": self.last_command.reason,
            "dynamic_tracks": len(self.dynamic_frame.tracks),
            "planning_tracks": len(self.dynamic_frame.planning_tracks),
            "active_tracks": self.dynamic_frame.active_track_count,
            "predicted_clearance": (
                None
                if not math.isfinite(self.safety.last_min_clearance)
                else round(self.safety.last_min_clearance, 3)
            ),
            "ttc": None if self.safety.last_ttc is None else round(self.safety.last_ttc, 3),
            "safety_override": self.safety.override_active,
            "safety_reason": self.safety.last_override_reason,
            "dynamic_state": [
                [
                    track.track_id,
                    round(track.x, 2),
                    round(track.y, 2),
                    round(track.vx, 2),
                    round(track.vy, 2),
                    round(track.confidence, 2),
                ]
                for track in self.dynamic_frame.tracks
            ],
        }
        self.emitter.send(json.dumps(status).encode("utf-8"))
        print(
            f"[STATUS] t={now:6.1f}s phase={status['phase']:<16} "
            f"pose=({pose.x:+.2f},{pose.y:+.2f},{pose.theta:+.2f}) "
            f"known={status['map_cells']:5d} target={status['target_confidence']:.2f}/"
            f"{status['target_range'] if status['target_range'] is not None else '-'}m "
            f"visited={status['targets_visited']}/{status['targets_required']} "
            f"dyn={status['dynamic_tracks']}/{status['planning_tracks']}/{status['active_tracks']} "
            f"ttc={status['ttc'] if status['ttc'] is not None else '-'} "
            f"safety={status['safety_reason'] or '-'} tracks={status['dynamic_state']} "
            f"cmd=({self.last_command.linear:+.2f},{self.last_command.angular:+.2f}) {self.last_command.reason}"
        )

    def run(self) -> None:
        if self.robot.step(self.time_step) == -1:
            return
        now = self.robot.getTime()
        self.mission.start(now, self.estimator.pose)
        print("[MISSION] autonomous stack online; start pose is the local-map origin")
        try:
            while self.robot.step(self.time_step) != -1:
                self.step_count += 1
                now = self.robot.getTime()
                left = self.left_sensor.getValue()
                right = self.right_sensor.getValue()
                gyro_values = self.gyro.getValues()
                self.estimator.update(left, right, float(gyro_values[2]), self.dt)
                scan = self._scan()
                if self.step_count % 16 == 0:
                    self.estimator.correct_with_scan(scan, self.grid)
                self.dynamic_frame = self.dynamic_tracker.update(
                    now,
                    self.estimator.pose,
                    scan,
                    self.grid,
                )
                self.grid.update(self.estimator.pose, scan, self.dynamic_frame.ignored_hit_mask)
                self.grid.decay_dynamic_regions(self.dynamic_frame.tracks)
                self.last_detection = self._detection(scan)

                transition = self.mission.update(now, self.estimator.pose, self.last_detection)
                if self.mission.visited_count > self.last_visited_count:
                    self.departure_until = now + 7.0
                    self.last_visited_count = self.mission.visited_count
                elif transition and "confirmed; resume search" in transition:
                    self.departure_until = now + 7.0
                if transition:
                    print(f"[MISSION] {transition}")
                    self.path = []
                    self.current_goal = None
                    self.fallback_heading = None
                    if self.mission.phase != MissionPhase.EXPLORE:
                        self.frontier_goal = None

                command = self._nominal_command(now, scan)
                command = self.safety.guard(
                    command,
                    scan,
                    self.estimator.pose,
                    tuple(self.dynamic_frame.safety_tracks),
                    now,
                )
                if (
                    self.mission.phase not in (MissionPhase.BOOTSTRAP, MissionPhase.RECOVERY, MissionPhase.COMPLETE)
                    and self.safety.is_stuck(now, self.estimator.pose, command)
                ):
                    self.mission.enter_recovery(now, "commanded motion without pose progress")
                    command = self.safety.guard(
                        ControlCommand(-0.05, 0.9, "begin stuck recovery"),
                        scan,
                        self.estimator.pose,
                        tuple(self.dynamic_frame.safety_tracks),
                        now,
                    )
                self.last_command = command
                self._actuate(command)

                if self.step_count % 32 == 0:
                    self.display.draw(
                        self.grid,
                        self.estimator.pose,
                        self.path,
                        self.current_goal,
                        self.dynamic_frame.tracks,
                    )
                self._publish(now)
        except Exception:
            self._actuate(ControlCommand())
            print("[FATAL] controller stopped safely after an exception")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    AutonomousSearchAndRescue().run()

