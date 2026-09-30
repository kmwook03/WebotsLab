import math
from pathlib import Path
import sys
import unittest

import numpy as np


CONTROLLER = Path(__file__).resolve().parents[1] / "controllers" / "amr_search_rescue"
sys.path.insert(0, str(CONTROLLER))

from config import DynamicObstacleConfig, MapConfig, MissionConfig, PlannerConfig, RobotConfig, SafetyConfig, TargetConfig, TraversabilityConfig  # noqa: E402
from collision import predict_dynamic_clearance  # noqa: E402
from mapping import OccupancyGrid  # noqa: E402
from localization import PoseEstimator  # noqa: E402
from mission import MissionManager  # noqa: E402
from models import ControlCommand, DynamicObstacle, LaserScan, MissionPhase, Pose2D, TargetDetection, TraversabilityFrame, Velocity, wrap_angle  # noqa: E402
from perception import RedTargetDetector, TraversableAreaDetector  # noqa: E402
from planning import AStarPlanner, DynamicWindowPlanner, FrontierExplorer, WaypointProgressMonitor, append_breadcrumb, breadcrumb_return_cost, choose_local_detour, reverse_breadcrumb_segment, terminal_approach_path  # noqa: E402
from safety import SafetySupervisor  # noqa: E402
from tracking import DynamicObstacleTracker  # noqa: E402


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

    def test_gyro_dominates_heading_when_wheels_slip(self):
        estimator = PoseEstimator(RobotConfig(gyro_weight=0.75))
        estimator.update(0.0, 0.0, 0.0, 0.1)
        estimator.update(-0.2, 0.2, 1.0, 0.1)

        wheel_rotation = 0.4 * 0.033 / 0.160
        expected = 0.25 * wheel_rotation + 0.75 * 0.1
        self.assertAlmostEqual(estimator.pose.theta, expected, places=6)


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

    def test_reached_frontier_is_temporarily_excluded(self):
        self.grid.log_odds.fill(-2.0)
        self.grid.log_odds[18:24, 18:24] = 0.0
        self.grid.log_odds[66:72, 66:72] = 0.0
        explorer = FrontierExplorer(self.plan_config, self.planner)
        first_goal, first_result = explorer.choose(self.grid, Pose2D())

        self.assertIsNotNone(first_goal)
        self.assertIsNotNone(first_result)
        second_goal, second_result = explorer.choose(
            self.grid,
            Pose2D(*first_goal),
            [first_goal],
        )

        self.assertIsNotNone(second_goal)
        self.assertIsNotNone(second_result)
        self.assertGreater(
            math.hypot(
                second_goal[0] - first_goal[0],
                second_goal[1] - first_goal[1],
            ),
            self.plan_config.frontier_exclusion_radius,
        )


class LocalPlanningTests(unittest.TestCase):
    def test_breadcrumbs_ignore_jitter_and_return_in_reverse_order(self):
        trail = []
        self.assertTrue(append_breadcrumb(trail, Pose2D(0.0, 0.0), 0.20))
        self.assertFalse(append_breadcrumb(trail, Pose2D(0.08, 0.02), 0.20))
        self.assertTrue(append_breadcrumb(trail, Pose2D(0.25, 0.0), 0.20))
        self.assertTrue(append_breadcrumb(trail, Pose2D(0.50, 0.15), 0.20))
        self.assertTrue(append_breadcrumb(trail, Pose2D(0.75, 0.30), 0.20))

        segment, target_index = reverse_breadcrumb_segment(
            trail, Pose2D(0.78, 0.31), None, 0.55
        )

        self.assertEqual(target_index, 1)
        self.assertEqual(segment[1:], [trail[2], trail[1]])

    def test_breadcrumbs_erase_closed_route_loop(self):
        trail = [(0.0, 0.0), (0.4, 0.0), (0.8, 0.0), (0.8, 0.4), (0.4, 0.4)]

        changed = append_breadcrumb(
            trail,
            Pose2D(0.05, 0.04),
            spacing=0.20,
            loop_rejoin_distance=0.15,
            loop_guard_points=2,
        )

        self.assertTrue(changed)
        self.assertEqual(trail, [(0.0, 0.0)])

    def test_breadcrumb_stall_counts_only_active_control_time(self):
        monitor = WaypointProgressMonitor(min_progress=0.08, timeout=1.0)
        monitor.reset(1.0)

        for _ in range(20):
            self.assertFalse(monitor.update(0.98, 0.1, active=False))
        for _ in range(9):
            self.assertFalse(monitor.update(0.97, 0.1, active=True))
        self.assertTrue(monitor.update(0.97, 0.1, active=True))
        self.assertFalse(monitor.update(0.86, 0.1, active=True))

    def test_breadcrumb_cost_exposes_excessive_remembered_detour(self):
        trail = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.2, 2.0)]

        cost = breadcrumb_return_cost(trail, Pose2D(0.2, 1.0))
        direct = math.hypot(0.2, 1.0)

        self.assertGreater(cost, 2.5 * direct + 0.5)

    def test_home_detour_prefers_open_side_of_blocked_direct_corridor(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, 3.5, dtype=np.float32)
        ranges[np.abs(angles) < math.radians(18.0)] = 0.35
        ranges[(angles < 0.0) & (angles > math.radians(-100.0))] = 0.45
        scan = LaserScan(ranges, angles, 0.05, 3.5)

        waypoint = choose_local_detour(
            Pose2D(),
            (1.2, 0.0),
            scan,
            RobotConfig(),
            0.70,
            0.30,
            math.radians(12.0),
            math.radians(110.0),
        )

        self.assertIsNotNone(waypoint)
        self.assertGreater(waypoint[1], 0.20)
        self.assertGreater(waypoint[0], 0.0)

    def test_home_detour_is_unused_when_direct_corridor_is_clear(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, 3.5, dtype=np.float32), angles, 0.05, 3.5)

        waypoint = choose_local_detour(
            Pose2D(),
            (1.2, 0.0),
            scan,
            RobotConfig(),
            0.70,
            0.30,
            math.radians(12.0),
            math.radians(110.0),
        )

        self.assertIsNone(waypoint)

    def test_forced_home_detour_penalizes_previously_visited_side(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, 3.5, dtype=np.float32)
        ranges[np.abs(angles) < math.radians(16.0)] = 0.35
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        visited_upper = [(0.25, 0.30), (0.45, 0.48), (0.65, 0.58)]

        waypoint = choose_local_detour(
            Pose2D(),
            (1.2, 0.0),
            scan,
            RobotConfig(),
            0.70,
            0.30,
            math.radians(12.0),
            math.radians(110.0),
            visited_upper,
            0.42,
            True,
        )

        self.assertIsNotNone(waypoint)
        self.assertLess(waypoint[1], -0.15)

    def test_return_terminal_path_rejects_far_snapped_endpoint(self):
        pose = Pose2D(0.96, 0.65, -0.19)

        path, direct = terminal_approach_path(
            [(0.96, 0.65)], pose, (0.0, 0.0), 1.80, 0.35
        )

        self.assertTrue(direct)
        self.assertEqual(path[-1], (0.0, 0.0))

    def test_return_terminal_path_preserves_valid_or_distant_global_path(self):
        near_pose = Pose2D(0.96, 0.65)
        valid_path = [(0.96, 0.65), (0.1, 0.1)]
        preserved, near_direct = terminal_approach_path(
            valid_path, near_pose, (0.0, 0.0), 1.80, 0.35
        )
        distant_path = [(3.0, 0.0), (1.0, 0.0)]
        distant, far_direct = terminal_approach_path(
            distant_path, Pose2D(3.0, 0.0), (0.0, 0.0), 1.80, 0.35
        )

        self.assertFalse(near_direct)
        self.assertEqual(preserved, valid_path)
        self.assertFalse(far_direct)
        self.assertEqual(distant, distant_path)

    def test_dwa_advances_after_direct_home_alignment(self):
        robot = RobotConfig()
        planner = DynamicWindowPlanner(robot, PlannerConfig(), DynamicObstacleConfig())
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5)
        pose = Pose2D(0.96, 0.65, -0.19)
        velocity = Velocity()
        initial_distance = math.hypot(pose.x, pose.y)
        max_linear = 0.0

        for index in range(120):
            command = planner.command(
                pose,
                velocity,
                [(pose.x, pose.y), (0.0, 0.0)],
                scan,
                0.032,
                (),
                index * 0.032,
            )
            velocity = Velocity(command.linear, command.angular)
            pose.theta += command.angular * 0.032
            pose.x += command.linear * math.cos(pose.theta) * 0.032
            pose.y += command.linear * math.sin(pose.theta) * 0.032
            max_linear = max(max_linear, command.linear)

        self.assertGreater(max_linear, 0.10)
        self.assertLess(math.hypot(pose.x, pose.y), initial_distance - 0.30)

    def test_dwa_detour_endpoint_is_not_pulled_toward_home_early(self):
        planner = DynamicWindowPlanner(
            RobotConfig(), PlannerConfig(), DynamicObstacleConfig()
        )
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(
            np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5
        )
        pose = Pose2D(0.2, 0.2, math.pi / 4.0)
        velocity = Velocity(0.12, 0.0)
        waypoint = (0.5, 0.5)
        home = (1.2, 0.0)

        corner_cutting = planner.command(
            pose,
            velocity,
            [(pose.x, pose.y), waypoint, home],
            scan,
            0.064,
            (),
            0.0,
        )
        local_goal = planner.command(
            pose,
            velocity,
            [(pose.x, pose.y), waypoint],
            scan,
            0.064,
            (),
            0.0,
        )

        self.assertLess(corner_cutting.angular, -0.10)
        self.assertGreater(local_goal.angular, -0.05)

    def test_dwa_rejects_collision_course(self):
        robot = RobotConfig()
        planner = DynamicWindowPlanner(robot, PlannerConfig(), DynamicObstacleConfig())
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.argmin(np.abs(angles))] = 0.26
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        command = planner.command(
            Pose2D(), Velocity(0.16, 0.0), [(0.0, 0.0), (2.0, 0.0)], scan, 0.064, (), 0.0
        )
        self.assertLessEqual(command.linear, 0.02)
        self.assertGreater(abs(command.angular), 0.1)

    def test_side_proximity_uses_scored_evasive_candidate(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, 3.5, dtype=np.float32)
        side_index = int(np.argmin(np.abs(angles - math.pi / 2.0)))
        ranges[side_index] = 0.18
        scan = LaserScan(ranges, angles, 0.12, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        command = supervisor.guard(ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 0.0)

        self.assertTrue(command.reason.startswith("predictive evasive"))
        self.assertNotEqual((command.linear, command.angular), (0.20, 0.0))

    def test_front_stop_latches_turn_toward_clearer_side(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, 3.5, dtype=np.float32)
        front = np.abs(angles) < math.radians(8.0)
        right = (angles < math.radians(-20.0)) & (angles > math.radians(-105.0))
        ranges[front] = 0.34
        ranges[right] = 0.55
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        initial = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 0.0
        )
        supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(np.full(360, 3.5, dtype=np.float32), angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.1,
        )
        supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 3.0
        )
        supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(np.full(360, 3.5, dtype=np.float32), angles, 0.05, 3.5),
            Pose2D(),
            (),
            3.1,
        )
        first = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 6.0
        )
        second = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 6.1
        )

        self.assertTrue(initial.reason.startswith("predictive"))
        self.assertEqual(first.reason, "front escape turn")
        self.assertEqual(second.reason, "front escape turn")
        self.assertEqual(first.linear, 0.0)
        self.assertGreater(first.angular, 0.0)
        self.assertEqual(first.angular, second.angular)

    def test_front_escape_turn_continues_before_short_release(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        blocked_ranges = np.full(360, 3.5, dtype=np.float32)
        blocked_ranges[np.abs(angles) < math.radians(8.0)] = 0.34
        blocked = LaserScan(blocked_ranges, angles, 0.05, 3.5)
        clear = LaserScan(np.full(360, 3.5, dtype=np.float32), angles, 0.05, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())
        nominal = ControlCommand(0.20, 0.0, "test")

        initial = supervisor.guard(nominal, blocked, Pose2D(), (), 0.0)
        supervisor.guard(nominal, clear, Pose2D(), (), 0.1)
        supervisor.guard(nominal, blocked, Pose2D(), (), 3.0)
        supervisor.guard(nominal, clear, Pose2D(), (), 3.1)
        first = supervisor.guard(nominal, blocked, Pose2D(), (), 6.0)
        still_turning = supervisor.guard(nominal, clear, Pose2D(), (), 6.2)
        dwell = supervisor.guard(nominal, clear, Pose2D(), (), 6.4)
        released = supervisor.guard(nominal, clear, Pose2D(), (), 6.6)

        self.assertTrue(initial.reason.startswith("predictive"))
        self.assertEqual(first.reason, "front escape turn")
        self.assertEqual(still_turning.reason, "front escape turn")
        self.assertEqual(dwell.reason, "safety release dwell")
        self.assertEqual(released.reason, "test")
        self.assertFalse(supervisor.override_active)

    def test_front_escape_does_not_activate_while_robot_changes_region(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        blocked_ranges = np.full(360, 3.5, dtype=np.float32)
        blocked_ranges[np.abs(angles) < math.radians(8.0)] = 0.34
        blocked = LaserScan(blocked_ranges, angles, 0.05, 3.5)
        clear = LaserScan(np.full(360, 3.5, dtype=np.float32), angles, 0.05, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())
        nominal = ControlCommand(0.20, 0.0, "test")

        commands = []
        for now, x in ((0.0, 0.0), (3.0, 0.30), (6.0, 0.60)):
            commands.append(supervisor.guard(nominal, blocked, Pose2D(x=x), (), now))
            supervisor.guard(nominal, clear, Pose2D(x=x), (), now + 0.1)

        self.assertTrue(
            all(command.reason.startswith("predictive") for command in commands)
        )
        self.assertTrue(all(command.reason != "front escape turn" for command in commands))

    def test_evasive_margin_changes_tight_dynamic_escape(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5)
        obstacle = DynamicObstacle(1, 0.35, -0.20, -0.40, 0.0, 0.16, 1.0, 0.0)
        loose = SafetySupervisor(
            RobotConfig(),
            SafetyConfig(),
            DynamicObstacleConfig(evasive_safety_margin=0.02),
        )
        strict = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        loose_command = loose._evasive_command(Pose2D(), scan, (obstacle,), 0.0)
        strict_command = strict._evasive_command(Pose2D(), scan, (obstacle,), 0.0)

        self.assertTrue(loose_command.reason.startswith("predictive evasive"))
        self.assertTrue(strict_command.reason.startswith("predictive evasive"))
        self.assertNotEqual(
            (loose_command.linear, loose_command.angular),
            (strict_command.linear, strict_command.angular),
        )

    def test_closing_person_inside_margin_uses_best_effort_escape(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        side = int(np.argmin(np.abs(angles - math.pi / 2.0)))
        ranges[side - 4 : side + 5] = 0.16
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        obstacle = DynamicObstacle(
            1, 0.0, 0.30, 0.0, -0.35, 0.16, 1.0, 0.0
        )
        robot = RobotConfig()
        dynamic = DynamicObstacleConfig()
        supervisor = SafetySupervisor(robot, SafetyConfig(), dynamic)

        command = supervisor._evasive_command(Pose2D(), scan, (obstacle,), 0.0)
        recovery_config = DynamicObstacleConfig(
            safety_margin=dynamic.evasive_safety_margin
        )
        escape = predict_dynamic_clearance(
            Pose2D(), command, (obstacle,), robot, recovery_config, 0.0
        )
        stopped = predict_dynamic_clearance(
            Pose2D(), ControlCommand(), (obstacle,), robot, recovery_config, 0.0
        )

        self.assertTrue(command.reason.startswith("predictive evasive"))
        self.assertGreater(escape.final_clearance, stopped.final_clearance)

    def test_pretrack_layer_stops_for_compact_untracked_closing_object(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        side = int(np.argmin(np.abs(angles - math.pi / 2.0)))
        previous_ranges = np.full(360, np.inf, dtype=np.float32)
        current_ranges = np.full(360, np.inf, dtype=np.float32)
        previous_ranges[side - 3:side + 4] = 0.60
        current_ranges[side - 3:side + 4] = 0.52
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        first = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(previous_ranges, angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.0,
            Velocity(0.20, 0.0),
        )
        second = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(current_ranges, angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.1,
            Velocity(0.20, 0.0),
        )
        third = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(
                current_ranges - np.where(np.isfinite(current_ranges), 0.08, 0.0),
                angles,
                0.05,
                3.5,
            ),
            Pose2D(),
            (),
            0.2,
            Velocity(0.20, 0.0),
        )

        self.assertEqual(first.reason, "test")
        self.assertEqual(second.reason, "test")
        self.assertIn("pretrack closing", supervisor.last_override_reason)
        self.assertEqual(third.reason, "pretrack safety stop")

    def test_pretrack_confirmation_requires_same_angular_region(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        left = int(np.argmin(np.abs(angles - math.pi / 2.0)))
        right = int(np.argmin(np.abs(angles + math.pi / 2.0)))
        baseline = np.full(360, np.inf, dtype=np.float32)
        first_change = baseline.copy()
        second_change = baseline.copy()
        baseline[left - 3:left + 4] = 0.60
        first_change[left - 3:left + 4] = 0.52
        first_change[right - 3:right + 4] = 0.60
        second_change[left - 3:left + 4] = 0.52
        second_change[right - 3:right + 4] = 0.52
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        commands = []
        for now, ranges in ((0.0, baseline), (0.1, first_change), (0.2, second_change)):
            commands.append(
                supervisor.guard(
                    ControlCommand(0.0, 0.0, "test"),
                    LaserScan(ranges, angles, 0.05, 3.5),
                    Pose2D(),
                    (),
                    now,
                    Velocity(),
                )
            )

        self.assertTrue(all(command.reason == "test" for command in commands))
        self.assertFalse(supervisor.override_active)

    def test_pretrack_layer_compensates_for_ego_approach_to_static_surface(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        front = int(np.argmin(np.abs(angles)))
        previous_ranges = np.full(360, np.inf, dtype=np.float32)
        current_ranges = np.full(360, np.inf, dtype=np.float32)
        previous_ranges[front - 3:front + 4] = 0.60
        current_ranges[front - 3:front + 4] = 0.58
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(previous_ranges, angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.0,
            Velocity(0.20, 0.0),
        )
        command = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(current_ranges, angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.1,
            Velocity(0.20, 0.0),
        )

        self.assertEqual(command.reason, "test")
        self.assertFalse(supervisor.override_active)

    def test_pretrack_layer_ignores_scan_change_during_rotation(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        side = int(np.argmin(np.abs(angles - math.pi / 2.0)))
        previous_ranges = np.full(360, np.inf, dtype=np.float32)
        current_ranges = np.full(360, np.inf, dtype=np.float32)
        previous_ranges[side - 3:side + 4] = 0.60
        current_ranges[side - 3:side + 4] = 0.52
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        supervisor.guard(
            ControlCommand(0.0, 0.50, "test"),
            LaserScan(previous_ranges, angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.0,
            Velocity(0.0, 0.50),
        )
        command = supervisor.guard(
            ControlCommand(0.0, 0.50, "test"),
            LaserScan(current_ranges, angles, 0.05, 3.5),
            Pose2D(),
            (),
            0.1,
            Velocity(0.0, 0.50),
        )

        self.assertEqual(command.reason, "test")
        self.assertFalse(supervisor.override_active)

    def test_side_person_moving_away_does_not_trigger_360_proximity(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.argmin(np.abs(angles - math.pi / 2.0))] = 0.30
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())
        receding = DynamicObstacle(1, 0.0, 0.46, 0.0, 0.35, 0.16, 1.0, 0.0)

        command = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (receding,), 0.0
        )

        self.assertEqual(command.reason, "test")
        self.assertFalse(supervisor.override_active)

    def test_rear_person_moving_away_does_not_trigger_360_proximity(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.argmin(np.abs(np.abs(angles) - math.pi))] = 0.30
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())
        receding = DynamicObstacle(1, -0.46, 0.0, -0.35, 0.0, 0.16, 1.0, 0.0)

        command = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (receding,), 0.0
        )

        self.assertEqual(command.reason, "test")
        self.assertFalse(supervisor.override_active)

    def test_non_closing_side_and_rear_people_do_not_trigger_360_proximity(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())

        for track_id, x, y, bearing in (
            (1, 0.0, 0.46, math.pi / 2.0),
            (2, -0.46, 0.0, math.pi),
        ):
            with self.subTest(track_id=track_id):
                ranges = np.full(360, np.inf, dtype=np.float32)
                ranges[np.argmin(np.abs(np.abs(angles) - abs(bearing)))] = 0.30
                scan = LaserScan(ranges, angles, 0.05, 3.5)
                stationary = DynamicObstacle(
                    track_id, x, y, 0.0, 0.0, 0.16, 1.0, 0.0
                )

                command = supervisor.guard(
                    ControlCommand(0.20, 0.0, "test"),
                    scan,
                    Pose2D(),
                    (stationary,),
                    0.0,
                )

                self.assertEqual(command.reason, "test")
                self.assertFalse(supervisor.override_active)

    def test_visible_receding_hazard_skips_lost_track_hold(self):
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.argmin(np.abs(angles - math.pi / 2.0))] = 0.30
        scan = LaserScan(ranges, angles, 0.05, 3.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), DynamicObstacleConfig())
        approaching = DynamicObstacle(1, 0.0, 0.46, 0.0, -0.35, 0.16, 1.0, 0.0)
        receding = DynamicObstacle(1, 0.0, 0.48, 0.0, 0.35, 0.16, 1.0, 0.1)

        first = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (approaching,), 0.0
        )
        release = supervisor.guard(
            ControlCommand(0.20, 0.0, "test"),
            LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5),
            Pose2D(),
            (receding,),
            0.1,
        )

        self.assertTrue(first.reason.startswith("predictive"))
        self.assertEqual(release.reason, "safety release dwell")
        self.assertEqual(supervisor.last_override_reason, "safety release dwell")

    def test_deliberate_rotation_is_not_reported_as_stuck(self):
        supervisor = SafetySupervisor(
            RobotConfig(),
            SafetyConfig(stuck_window=1.0, stuck_distance=0.08),
            DynamicObstacleConfig(),
        )
        command = ControlCommand(0.0, 0.9, "scan")

        for now, theta in ((0.0, 0.0), (0.4, 0.3), (0.8, 0.6), (1.0, 0.9)):
            self.assertFalse(supervisor.is_stuck(now, Pose2D(theta=theta), command))

    def test_safety_override_does_not_feed_stuck_recovery(self):
        supervisor = SafetySupervisor(
            RobotConfig(),
            SafetyConfig(stuck_window=1.0, stuck_distance=0.08),
            DynamicObstacleConfig(),
        )
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5)
        crossing = DynamicObstacle(1, 0.50, -0.40, 0.0, 0.55, 0.16, 1.0, 0.0)
        pose = Pose2D()

        for now in (0.0, 0.4, 0.8, 1.2, 1.6):
            guarded = supervisor.guard(
                ControlCommand(0.20, 0.0, "DWA"),
                scan,
                pose,
                (crossing,),
                now,
            )
            self.assertFalse(supervisor.is_stuck(now, pose, guarded))
            self.assertEqual(len(supervisor.history), 0)

    def test_intentional_stop_starts_a_fresh_stuck_window(self):
        supervisor = SafetySupervisor(
            RobotConfig(),
            SafetyConfig(stuck_window=1.0, stuck_distance=0.08),
            DynamicObstacleConfig(),
        )
        pose = Pose2D()

        self.assertFalse(supervisor.is_stuck(0.0, pose, ControlCommand(0.20, 0.0, "DWA")))
        self.assertFalse(supervisor.is_stuck(0.4, pose, ControlCommand(0.20, 0.0, "DWA")))
        self.assertGreater(len(supervisor.history), 0)

        self.assertFalse(supervisor.is_stuck(0.8, pose, ControlCommand(0.0, 0.0, "confirm target")))
        self.assertEqual(len(supervisor.history), 0)

        # A departure immediately after the intentional stop begins with a
        # clean timer instead of inheriting the preceding stationary samples.
        self.assertFalse(supervisor.is_stuck(1.0, pose, ControlCommand(-0.18, 0.16, "leave target")))

    def test_dwa_rejects_predicted_crossing(self):
        dynamic_config = DynamicObstacleConfig()
        planner = DynamicWindowPlanner(RobotConfig(), PlannerConfig(), dynamic_config)
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5)
        crossing = DynamicObstacle(1, 0.55, -0.48, 0.0, 0.52, 0.16, 1.0, 0.0)

        command = planner.command(
            Pose2D(),
            Velocity(0.20, 0.0),
            [(0.0, 0.0), (2.0, 0.0)],
            scan,
            0.064,
            (crossing,),
            0.0,
        )

        self.assertTrue(command.linear < 0.15 or abs(command.angular) > 0.15)

    def test_dwa_anticipates_crossing_beyond_rollout_horizon(self):
        dynamic_config = DynamicObstacleConfig(planning_prediction_horizon=2.2)
        planner = DynamicWindowPlanner(RobotConfig(), PlannerConfig(), dynamic_config)
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5)
        crossing = DynamicObstacle(1, 0.40, -1.0, 0.0, 0.50, 0.16, 0.55, 0.0)

        command = planner.command(
            Pose2D(),
            Velocity(0.20, 0.0),
            [(0.0, 0.0), (2.0, 0.0)],
            scan,
            0.064,
            (crossing,),
            0.0,
        )

        self.assertGreater(abs(command.angular), 0.15)

    def test_prediction_advances_track_across_short_occlusion(self):
        obstacle = DynamicObstacle(1, 0.90, 0.0, -0.50, 0.0, 0.16, 1.0, 0.0)
        prediction = predict_dynamic_clearance(
            Pose2D(),
            ControlCommand(0.0, 0.0, "wait"),
            (obstacle,),
            RobotConfig(),
            DynamicObstacleConfig(prediction_horizon=0.8, prediction_dt=0.1),
            now=0.6,
        )

        self.assertIsNotNone(prediction.ttc)
        self.assertLessEqual(prediction.ttc, 0.3)

    def test_predictive_guard_latches_until_release_dwell(self):
        dynamic_config = DynamicObstacleConfig(release_dwell=0.5)
        supervisor = SafetySupervisor(RobotConfig(), SafetyConfig(), dynamic_config)
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        scan = LaserScan(np.full(360, np.inf, dtype=np.float32), angles, 0.05, 3.5)
        crossing = DynamicObstacle(1, 0.50, -0.40, 0.0, 0.55, 0.16, 1.0, 0.0)

        first = supervisor.guard(ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (crossing,), 0.0)
        dwell = supervisor.guard(ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 0.2)
        still_held = supervisor.guard(ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 1.1)
        released = supervisor.guard(ControlCommand(0.20, 0.0, "test"), scan, Pose2D(), (), 1.7)

        self.assertIn("predictive", first.reason)
        self.assertEqual(dwell.reason, "predictive safety stop")
        self.assertEqual(still_held.reason, "safety release dwell")
        self.assertEqual(released.reason, "test")


class DynamicObstacleTrackingTests(unittest.TestCase):
    @staticmethod
    def _cluster_scan(x: float | None, y: float = 0.0) -> LaserScan:
        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        if x is not None:
            bearing = math.atan2(y, x)
            distance = math.hypot(x, y)
            center = int(np.argmin(np.abs(angles - bearing)))
            ranges[max(0, center - 4) : min(360, center + 5)] = distance
        return LaserScan(ranges, angles, 0.05, 3.5)

    def test_translating_compact_cluster_is_confirmed(self):
        tracker = DynamicObstacleTracker(DynamicObstacleConfig())
        frame = None
        samples = ((0.0, -0.12), (0.1, -0.08), (0.2, -0.04), (0.3, 0.0))
        for index, (now, y) in enumerate(samples):
            frame = tracker.update(now, Pose2D(), self._cluster_scan(1.0, y))
            if index == 1:
                self.assertEqual(len(frame.tracks), 0)
                self.assertEqual(len(frame.planning_tracks), 1)
                self.assertGreater(frame.planning_tracks[0].vy, 0.15)

        self.assertEqual(len(frame.tracks), 1)
        self.assertEqual(len(frame.planning_tracks), 1)
        self.assertGreater(frame.tracks[0].vy, 0.15)
        self.assertTrue(np.any(frame.ignored_hit_mask))

        # Once confirmed, a brief pause must not remove the obstacle from DWA.
        paused = tracker.update(0.4, Pose2D(), self._cluster_scan(1.0, 0.0))
        self.assertEqual(len(paused.tracks), 1)
        self.assertEqual(len(paused.planning_tracks), 1)

    def test_stationary_cluster_and_ego_motion_are_not_dynamic(self):
        config = DynamicObstacleConfig()
        stationary = DynamicObstacleTracker(config)
        for now in (0.0, 0.1, 0.2, 0.3):
            frame = stationary.update(now, Pose2D(), self._cluster_scan(1.0))
        self.assertEqual(len(frame.tracks), 0)
        self.assertEqual(len(frame.planning_tracks), 0)

        compensated = DynamicObstacleTracker(config)
        for now, robot_x in ((0.0, 0.0), (0.1, 0.05), (0.2, 0.10), (0.3, 0.15)):
            frame = compensated.update(now, Pose2D(x=robot_x), self._cluster_scan(1.0 - robot_x))
        self.assertEqual(len(frame.tracks), 0)
        self.assertEqual(len(frame.planning_tracks), 0)

    def test_confirmed_track_survives_short_occlusion_then_expires(self):
        tracker = DynamicObstacleTracker(DynamicObstacleConfig(track_timeout=0.5))
        for now, y in ((0.0, -0.12), (0.1, -0.08), (0.2, -0.04), (0.3, 0.0)):
            tracker.update(now, Pose2D(), self._cluster_scan(1.0, y))

        short_gap = tracker.update(0.5, Pose2D(), self._cluster_scan(None))
        expired = tracker.update(1.0, Pose2D(), self._cluster_scan(None))

        self.assertEqual(len(short_gap.tracks), 1)
        self.assertEqual(len(expired.tracks), 0)

    def test_dynamic_hit_is_not_committed_to_static_map(self):
        grid = OccupancyGrid(MapConfig(size=80, resolution=0.1, ray_stride=1))
        scan = LaserScan(
            ranges=np.array([1.0], dtype=np.float32),
            angles=np.array([0.0], dtype=np.float32),
            min_range=0.05,
            max_range=4.0,
        )
        grid.update(Pose2D(), scan, np.array([True]))
        hit_x, hit_y = grid.world_to_grid(1.0, 0.0)

        self.assertLessEqual(grid.log_odds[hit_y, hit_x], 0.0)


class PerceptionAndMissionTests(unittest.TestCase):
    def test_traversable_floor_segmentation_and_lidar_alignment(self):
        height, width = 96, 160
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        image[:40, :, :3] = (45, 45, 45)
        image[40:, :, :3] = (178, 185, 180)
        image[54:, 66:94, :3] = (28, 58, 105)

        angles = np.linspace(math.pi, -math.pi, 360, dtype=np.float32)
        ranges = np.full(360, np.inf, dtype=np.float32)
        ranges[np.abs(angles) < math.radians(3.0)] = 0.82
        scan = LaserScan(ranges, angles, 0.05, 3.5)

        frame = TraversableAreaDetector().detect(
            image.tobytes(), width, height, 1.1, scan
        )
        centre = int(np.argmin(np.abs(frame.bearings)))
        side = int(np.argmin(np.abs(frame.bearings - 0.38)))

        self.assertLess(frame.confidence[centre], 0.15)
        self.assertGreater(frame.confidence[side], 0.80)
        self.assertTrue(frame.lidar_hits[centre])
        self.assertAlmostEqual(frame.lidar_ranges[centre], 0.82, delta=0.03)
        self.assertFalse(frame.lidar_hits[side])
        self.assertAlmostEqual(frame.lidar_ranges[side], scan.max_range)

    def test_visual_traversability_cost_needs_lidar_corroboration(self):
        frame = TraversabilityFrame(
            bearings=np.array([0.4, 0.0, -0.4], dtype=np.float32),
            confidence=np.array([1.0, 0.1, 0.1], dtype=np.float32),
            lidar_ranges=np.array([3.5, 0.8, 3.5], dtype=np.float32),
            lidar_hits=np.array([False, True, False]),
        )

        centre_cost = DynamicWindowPlanner._visual_traversability_cost(
            frame, 0.0, 0.6
        )
        colour_only_cost = DynamicWindowPlanner._visual_traversability_cost(
            frame, -0.4, 0.6
        )
        clear_cost = DynamicWindowPlanner._visual_traversability_cost(
            frame, 0.4, 0.6
        )

        self.assertGreater(centre_cost, 0.65)
        self.assertLess(colour_only_cost, 0.15)
        self.assertEqual(clear_cost, 0.0)

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

    def test_revisited_target_is_rejected_when_lidar_range_is_wrong(self):
        config = MissionConfig(
            bootstrap_rotation=1.0,
            target_stable_frames=2,
            required_target_count=2,
            target_dedup_distance=0.6,
            target_rearm_distance=1.0,
            target_dedup_bearing_deg=14.0,
            target_confirm_time=0.1,
        )
        mission = MissionManager(config)
        mission.start(0.0, Pose2D())
        mission.update(0.5, Pose2D(theta=0.6), TargetDetection())
        mission.update(1.0, Pose2D(theta=1.2), TargetDetection())

        first_pose = Pose2D(theta=0.0)
        seen = TargetDetection(seen=True, confidence=0.9, range_m=1.2)
        close = TargetDetection(seen=True, confidence=0.9, range_m=0.45)
        mission.update(1.1, first_pose, seen)
        mission.update(1.2, first_pose, seen)
        mission.update(1.3, first_pose, close)
        mission.update(1.5, first_pose, close)
        self.assertEqual(mission.phase, MissionPhase.EXPLORE)
        self.assertEqual(mission.visited_count, 1)

        # The same target is straight ahead after turning around, but an
        # unrelated LiDAR return makes it appear several metres farther away.
        # Bearing-based identity must prevent a second approach.
        revisit_pose = Pose2D(x=1.8, theta=math.pi)
        wrong_range = TargetDetection(seen=True, confidence=0.9, range_m=3.0)
        mission.update(1.6, revisit_pose, wrong_range)
        mission.update(1.7, revisit_pose, wrong_range)
        mission.update(1.8, revisit_pose, wrong_range)

        self.assertEqual(mission.phase, MissionPhase.EXPLORE)
        self.assertEqual(mission.visited_count, 1)

    def test_distinct_target_still_enters_approach_after_rearming(self):
        config = MissionConfig(
            bootstrap_rotation=1.0,
            target_stable_frames=2,
            required_target_count=2,
            target_dedup_distance=0.6,
            target_rearm_distance=1.0,
            target_confirm_time=0.1,
        )
        mission = MissionManager(config)
        mission.start(0.0, Pose2D())
        mission.update(0.5, Pose2D(theta=0.6), TargetDetection())
        mission.update(1.0, Pose2D(theta=1.2), TargetDetection())

        seen = TargetDetection(seen=True, confidence=0.9, range_m=1.2)
        close = TargetDetection(seen=True, confidence=0.9, range_m=0.45)
        first_pose = Pose2D(theta=0.0)
        mission.update(1.1, first_pose, seen)
        mission.update(1.2, first_pose, seen)
        mission.update(1.3, first_pose, close)
        mission.update(1.5, first_pose, close)

        second_pose = Pose2D(x=2.0, y=1.5, theta=0.0)
        mission.update(1.6, second_pose, seen)
        mission.update(1.7, second_pose, seen)

        self.assertEqual(mission.phase, MissionPhase.TARGET_APPROACH)
        self.assertEqual(mission.visited_count, 1)

        locked_target = mission.target_object_estimate
        distractor = TargetDetection(
            seen=True,
            confidence=0.9,
            bearing=math.radians(48.0),
            range_m=1.2,
        )
        mission.update(1.75, second_pose, distractor)
        self.assertFalse(mission.target_observation_accepted)
        self.assertEqual(mission.target_object_estimate, locked_target)

        revisit_pose = Pose2D(x=1.8, theta=math.pi)
        mission.update(1.8, revisit_pose, seen)

        # Seeing the old marker while a different candidate is already locked
        # must not discard or overwrite the in-progress target.
        self.assertEqual(mission.phase, MissionPhase.TARGET_APPROACH)
        self.assertEqual(mission.target_object_estimate, locked_target)

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

