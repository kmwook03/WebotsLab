import math
from pathlib import Path
import sys
import unittest

import numpy as np


CONTROLLER = Path(__file__).resolve().parents[1] / "controllers" / "amr_search_rescue"
sys.path.insert(0, str(CONTROLLER))

from config import MapConfig, MissionConfig, PlannerConfig, RobotConfig, SafetyConfig, TargetConfig  # noqa: E402
from mapping import OccupancyGrid  # noqa: E402
from mission import MissionManager  # noqa: E402
from models import ControlCommand, LaserScan, MissionPhase, Pose2D, TargetDetection, Velocity, wrap_angle  # noqa: E402
from perception import RedTargetDetector  # noqa: E402
from planning import AStarPlanner, DynamicWindowPlanner, FrontierExplorer  # noqa: E402
from safety import SafetySupervisor  # noqa: E402


class GeometryAndMappingTests(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-0.25), -0.25)

    def test_log_odds_ray_update(self):
        grid = OccupancyGrid(MapConfig(size=80, resolution=0.1, ray_stride=1))
        scan = LaserScan(
            ranges=np.array([1.0], dtype=np.float32),
            angles=np.array([0.0], dtype=np.float32),
            min_range=0.05,
            max_range=4.0,
        )
        for _ in range(3):
            grid.update(Pose2D(), scan)
        free_x, free_y = grid.world_to_grid(0.5, 0.0)
        hit_x, hit_y = grid.world_to_grid(1.0, 0.0)
        self.assertLess(grid.log_odds[free_y, free_x], 0.0)
        self.assertGreater(grid.log_odds[hit_y, hit_x], 0.0)


class GlobalPlanningTests(unittest.TestCase):
    def setUp(self):
        self.map_config = MapConfig(size=90, resolution=0.1)
        self.plan_config = PlannerConfig(inflation_radius=0.10, frontier_min_cells=4)
        self.grid = OccupancyGrid(self.map_config)
        self.planner = AStarPlanner(self.plan_config)

    def test_astar_routes_through_gap(self):
        self.grid.log_odds.fill(-2.0)
        self.grid.log_odds[8:82, 45] = 3.0
        self.grid.log_odds[60:74, 45] = -2.0
        result = self.planner.plan(self.grid, (-2.5, 0.0), (2.5, 0.0))
        self.assertIsNotNone(result)
        self.assertGreater(len(result.path), 2)
        self.assertTrue(any(y > 1.0 for x, y in result.path if abs(x) < 0.3))

    def test_frontier_selection_is_reachable(self):
        self.grid.log_odds[30:61, 30:61] = -2.0
        explorer = FrontierExplorer(self.plan_config, self.planner)
        goal, result = explorer.choose(self.grid, Pose2D())
        self.assertIsNotNone(goal)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.path), 0)


class LocalPlanningTests(unittest.TestCase):
    def test_dwa_rejects_collision_course(self):
        robot = RobotConfig()
        planner = DynamicWindowPlanner(robot, PlannerConfig())
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.argmin(np.abs(angles))] = 0.26
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        command = planner.command(
            Pose2D(), Velocity(0.16, 0.0), [(0.0, 0.0), (2.0, 0.0)], scan, 0.064
        )
        self.assertLessEqual(command.linear, 0.02)
        self.assertGreater(abs(command.angular), 0.1)

    def test_side_proximity_triggers_360_escape(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, 3.5, dtype=np.float32)
        side_index = int(np.argmin(np.abs(angles - math.pi / 2.0)))
        ranges[side_index] = 0.18
        scan = LaserScan(ranges, angles, 0.12, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig())

        command = supervisor.guard(ControlCommand(0.20, 0.0, "test"), scan)

        self.assertEqual(command.reason, "360 proximity escape")
        self.assertLess(command.angular, 0.0)

    def test_deliberate_rotation_is_not_reported_as_stuck(self):
        supervisor = SafetySupervisor(
            RobotConfig(), SafetyConfig(stuck_window=1.0, stuck_distance=0.08)
        )
        command = ControlCommand(0.0, 0.9, "scan")

        for now, theta in ((0.0, 0.0), (0.4, 0.3), (0.8, 0.6), (1.0, 0.9)):
            self.assertFalse(supervisor.is_stuck(now, Pose2D(theta=theta), command))


class PerceptionAndMissionTests(unittest.TestCase):
    def test_red_target_and_lidar_association(self):
        height, width = 96, 160
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[25:75, 70:90, 2] = 255
        image[25:75, 70:90, 3] = 255
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.abs(angles) < math.radians(4.0)] = 1.7
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        detection = RedTargetDetector().detect(image.tobytes(), width, height, 1.1, scan)
        self.assertTrue(detection.seen)
        self.assertAlmostEqual(detection.bearing, 0.0, delta=0.03)
        self.assertAlmostEqual(detection.range_m, 1.7, delta=0.12)

    def test_inconsistent_lidar_return_is_rejected(self):
        height, width = 96, 160
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[38:58, 74:86, 2] = 255
        image[38:58, 74:86, 3] = 255
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.abs(angles) < math.radians(4.0)] = 0.4
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        detection = RedTargetDetector().detect(image.tobytes(), width, height, 1.1, scan)
        self.assertGreater(detection.range_m, 2.0)

    def test_target_colour_profile_is_configurable(self):
        height, width = 96, 160
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[25:75, 70:90, 0] = 255
        image[25:75, 70:90, 3] = 255
        detector = RedTargetDetector(
            TargetConfig(primary_channel=0, secondary_channels=(1, 2))
        )

        detection = detector.detect(image.tobytes(), width, height, 1.1)

        self.assertTrue(detection.seen)
        self.assertAlmostEqual(detection.bearing, 0.0, delta=0.03)

    def test_complete_mission_state_sequence(self):
        config = MissionConfig(
            bootstrap_rotation=1.0,
            target_stable_frames=2,
            required_target_count=1,
            target_confirm_time=0.1,
        )
        mission = MissionManager(config)
        mission.start(0.0, Pose2D())
        mission.update(0.5, Pose2D(theta=0.6), TargetDetection())
        mission.update(1.0, Pose2D(theta=1.2), TargetDetection())
        self.assertEqual(mission.phase, MissionPhase.EXPLORE)
        seen = TargetDetection(seen=True, confidence=0.9, range_m=1.2)
        mission.update(1.1, Pose2D(theta=1.2), seen)
        mission.update(1.2, Pose2D(theta=1.2), seen)
        self.assertEqual(mission.phase, MissionPhase.TARGET_APPROACH)
        close = TargetDetection(seen=True, confidence=0.9, range_m=0.45)
        mission.update(1.3, Pose2D(theta=1.2), close)
        self.assertEqual(mission.phase, MissionPhase.CONFIRM_TARGET)
        mission.update(1.5, Pose2D(x=0.6), close)
        self.assertEqual(mission.phase, MissionPhase.RETURN_HOME)
        mission.update(1.6, Pose2D(), TargetDetection())
        self.assertEqual(mission.phase, MissionPhase.COMPLETE)

    def test_multiple_targets_are_visited_before_return(self):
        config = MissionConfig(
            bootstrap_rotation=1.0,
            target_stable_frames=2,
            required_target_count=2,
            target_dedup_distance=0.8,
            target_confirm_time=0.1,
        )
        mission = MissionManager(config)
        mission.start(0.0, Pose2D())
        mission.update(0.5, Pose2D(theta=0.6), TargetDetection())
        mission.update(1.0, Pose2D(theta=1.2), TargetDetection())
        seen = TargetDetection(seen=True, confidence=0.9, range_m=1.2)
        close = TargetDetection(seen=True, confidence=0.9, range_m=0.45)

        mission.update(1.1, Pose2D(theta=1.2), seen)
        mission.update(1.2, Pose2D(theta=1.2), seen)
        mission.update(1.3, Pose2D(theta=1.2), close)
        mission.update(1.5, Pose2D(theta=1.2), close)
        self.assertEqual(mission.phase, MissionPhase.EXPLORE)
        self.assertEqual(mission.visited_count, 1)

        mission.update(1.55, Pose2D(theta=1.2), seen)
        mission.update(1.58, Pose2D(theta=1.2), seen)
        self.assertEqual(mission.phase, MissionPhase.EXPLORE)
        self.assertEqual(mission.visited_count, 1)

        second_pose = Pose2D(x=3.0, theta=1.2)
        mission.update(1.6, second_pose, seen)
        mission.update(1.7, second_pose, seen)
        mission.update(1.8, second_pose, close)
        mission.update(2.0, second_pose, close)
        self.assertEqual(mission.phase, MissionPhase.RETURN_HOME)
        self.assertEqual(mission.visited_count, 2)

    def test_lost_target_returns_to_exploration(self):
        config = MissionConfig(
            bootstrap_rotation=1.0,
            target_stable_frames=2,
            required_target_count=2,
            target_reacquire_timeout=2.0,
        )
        mission = MissionManager(config)
        mission.start(0.0, Pose2D())
        mission.update(0.5, Pose2D(theta=0.6), TargetDetection())
        mission.update(1.0, Pose2D(theta=1.2), TargetDetection())
        seen = TargetDetection(seen=True, confidence=0.9, range_m=2.5)
        mission.update(1.1, Pose2D(theta=1.2), seen)
        mission.update(1.2, Pose2D(theta=1.2), seen)
        self.assertEqual(mission.phase, MissionPhase.TARGET_APPROACH)

        mission.update(3.3, Pose2D(x=-2.0), TargetDetection())

        self.assertEqual(mission.phase, MissionPhase.EXPLORE)


if __name__ == "__main__":
    unittest.main()

