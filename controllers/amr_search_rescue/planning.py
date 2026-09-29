"""Frontier exploration, A* global planning, and DWA local planning."""

from collections import deque
import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from collision import predict_dynamic_clearance
from config import DynamicObstacleConfig, PlannerConfig, RobotConfig
from mapping import GridCell, OccupancyGrid
from models import (
    ControlCommand,
    DynamicObstacle,
    LaserScan,
    PlanResult,
    Pose2D,
    Velocity,
    angle_difference,
    transform_points,
)


WorldPoint = Tuple[float, float]


def append_breadcrumb(
    trail: List[WorldPoint],
    pose: Pose2D,
    spacing: float,
    loop_rejoin_distance: float = 0.0,
    loop_guard_points: int = 3,
) -> bool:
    """Append pose progress and erase the tail when a route loop closes."""
    point = (pose.x, pose.y)
    if trail and math.hypot(point[0] - trail[-1][0], point[1] - trail[-1][1]) < spacing:
        return False
    searchable_count = max(0, len(trail) - max(1, loop_guard_points))
    if loop_rejoin_distance > 0.0 and searchable_count:
        rejoin_index = min(
            range(searchable_count),
            key=lambda index: math.hypot(
                point[0] - trail[index][0], point[1] - trail[index][1]
            ),
        )
        rejoin_distance = math.hypot(
            point[0] - trail[rejoin_index][0],
            point[1] - trail[rejoin_index][1],
        )
        if rejoin_distance <= loop_rejoin_distance:
            del trail[rejoin_index + 1 :]
            if rejoin_distance >= spacing:
                trail.append(point)
            return True
    trail.append(point)
    return True


def reverse_breadcrumb_segment(
    trail: Sequence[WorldPoint],
    pose: Pose2D,
    next_index: Optional[int],
    lookahead_distance: float,
) -> Tuple[List[WorldPoint], Optional[int]]:
    """Return a recent-to-old path segment and its oldest target index."""
    if len(trail) < 2:
        return [], None
    cursor = len(trail) - 2 if next_index is None else min(next_index, len(trail) - 1)
    if cursor < 0:
        return [], None

    path = [(pose.x, pose.y)]
    previous = path[0]
    travelled = 0.0
    target_index = cursor
    while target_index >= 0:
        point = trail[target_index]
        travelled += math.hypot(point[0] - previous[0], point[1] - previous[1])
        path.append(point)
        previous = point
        if travelled >= lookahead_distance or target_index == 0:
            break
        target_index -= 1
    return path, target_index


def breadcrumb_return_cost(
    trail: Sequence[WorldPoint],
    pose: Pose2D,
    next_index: Optional[int] = None,
) -> float:
    """Length of the remembered route from the current pose back to home."""
    if len(trail) < 2:
        return math.inf
    cursor = len(trail) - 2 if next_index is None else min(next_index, len(trail) - 1)
    if cursor < 0:
        return 0.0
    cost = math.hypot(pose.x - trail[cursor][0], pose.y - trail[cursor][1])
    for index in range(cursor, 0, -1):
        cost += math.hypot(
            trail[index][0] - trail[index - 1][0],
            trail[index][1] - trail[index - 1][1],
        )
    return cost


class WaypointProgressMonitor:
    """Measure commanded, non-safety time without meaningful target progress."""

    def __init__(self, min_progress: float, timeout: float):
        self.min_progress = min_progress
        self.timeout = timeout
        self.best_distance = math.inf
        self.active_elapsed = 0.0

    def reset(self, distance: float = math.inf) -> None:
        self.best_distance = distance
        self.active_elapsed = 0.0

    def update(self, distance: float, dt: float, active: bool) -> bool:
        if not math.isfinite(self.best_distance):
            self.reset(distance)
            return False
        if distance <= self.best_distance - self.min_progress:
            self.reset(distance)
            return False
        self.best_distance = min(self.best_distance, distance)
        if active:
            self.active_elapsed += max(0.0, dt)
        if self.active_elapsed + 1e-9 < self.timeout:
            return False
        self.reset(distance)
        return True


def terminal_approach_path(
    path: Sequence[WorldPoint],
    pose: Pose2D,
    requested_goal: WorldPoint,
    direct_approach_distance: float,
    max_endpoint_error: float,
) -> Tuple[List[WorldPoint], bool]:
    """Replace a badly snapped terminal path with a short live-sensor approach.

    A* may move a blocked requested goal to the nearest mapped free cell.  That
    is useful during ordinary navigation, but near the known start pose it can
    make the local planner stop at the snapped cell while the mission is still
    waiting for the true home tolerance.  The direct segment remains subject
    to DWA's live LiDAR collision checks and the independent safety guard.
    """
    planned = list(path)
    goal_distance = math.hypot(
        requested_goal[0] - pose.x,
        requested_goal[1] - pose.y,
    )
    endpoint_error = (
        math.hypot(
            planned[-1][0] - requested_goal[0],
            planned[-1][1] - requested_goal[1],
        )
        if planned
        else math.inf
    )
    if (
        goal_distance <= direct_approach_distance
        and endpoint_error > max_endpoint_error
    ):
        return [(pose.x, pose.y), requested_goal], True
    return planned, False


def choose_local_detour(
    pose: Pose2D,
    requested_goal: WorldPoint,
    scan: LaserScan,
    robot: RobotConfig,
    waypoint_distance: float,
    min_travel: float,
    corridor_half_angle: float,
    max_turn: float,
    visited_points: Sequence[WorldPoint] = (),
    visited_radius: float = 0.0,
    force: bool = False,
) -> Optional[WorldPoint]:
    """Choose a short, open LiDAR corridor that still progresses home."""
    goal_dx = requested_goal[0] - pose.x
    goal_dy = requested_goal[1] - pose.y
    goal_distance = math.hypot(goal_dx, goal_dy)
    if goal_distance <= 1e-6 or scan.ranges.size == 0:
        return None

    direct_bearing = angle_difference(math.atan2(goal_dy, goal_dx), pose.theta)
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    usable = np.where(
        np.isfinite(ranges),
        np.clip(ranges, 0.0, scan.max_range),
        scan.max_range,
    )
    if scan.angles.size > 1:
        angular_resolution = float(np.median(np.abs(np.diff(scan.angles))))
    else:
        angular_resolution = 2.0 * math.pi
    window_radius = max(1, int(round(corridor_half_angle / angular_resolution)))
    windows = [np.roll(usable, offset) for offset in range(-window_radius, window_radius + 1)]
    corridor_clearance = np.percentile(np.stack(windows), 20.0, axis=0)

    bearing_error = np.abs(
        (scan.angles - direct_bearing + math.pi) % (2.0 * math.pi) - math.pi
    )
    direct_index = int(np.argmin(bearing_error))
    footprint_clearance = robot.robot_radius + robot.safety_margin + 0.05
    required_direct = min(goal_distance, waypoint_distance) + footprint_clearance
    if not force and float(corridor_clearance[direct_index]) >= required_direct:
        return None

    best: Tuple[float, WorldPoint] | None = None
    for index in range(0, scan.angles.size, 3):
        bearing = float(scan.angles[index])
        turn_from_goal = abs(angle_difference(bearing, direct_bearing))
        if turn_from_goal > max_turn:
            continue
        available_travel = float(corridor_clearance[index]) - footprint_clearance
        travel = min(waypoint_distance, available_travel)
        if travel < min_travel:
            continue
        heading = pose.theta + bearing
        waypoint = (
            pose.x + travel * math.cos(heading),
            pose.y + travel * math.sin(heading),
        )
        remaining = math.hypot(
            requested_goal[0] - waypoint[0],
            requested_goal[1] - waypoint[1],
        )
        progress = goal_distance - remaining
        revisit_penalty = 0.0
        if visited_points and visited_radius > 0.0:
            nearest_visited = min(
                math.hypot(waypoint[0] - point[0], waypoint[1] - point[1])
                for point in visited_points
            )
            revisit_penalty = max(0.0, 1.0 - nearest_visited / visited_radius)
        score = (
            2.4 * progress
            + 0.35 * min(float(corridor_clearance[index]), 1.5)
            - 0.18 * turn_from_goal
            - 0.85 * revisit_penalty
        )
        if best is None or score > best[0]:
            best = (score, waypoint)
    return None if best is None else best[1]


class AStarPlanner:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self._moves = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )

    def plan(self, grid: OccupancyGrid, start: WorldPoint, goal: WorldPoint) -> Optional[PlanResult]:
        start_cell = grid.nearest_free(grid.world_to_grid(*start), max_radius=8)
        goal_cell = grid.nearest_free(grid.world_to_grid(*goal), max_radius=80)
        if start_cell is None or goal_cell is None:
            return None

        inflated = grid.inflated_obstacles(self.config.inflation_radius)
        blocked = inflated | grid.unknown_mask()
        blocked[start_cell[1], start_cell[0]] = False
        blocked[goal_cell[1], goal_cell[0]] = False

        size = grid.size
        costs = np.full((size, size), np.inf, dtype=np.float32)
        parents_x = np.full((size, size), -1, dtype=np.int16)
        parents_y = np.full((size, size), -1, dtype=np.int16)
        costs[start_cell[1], start_cell[0]] = 0.0
        queue = [(0.0, 0.0, start_cell[0], start_cell[1])]
        expanded = 0

        while queue:
            _, cost, x, y = heapq.heappop(queue)
            if cost > float(costs[y, x]) + 1e-5:
                continue
            expanded += 1
            if (x, y) == goal_cell:
                cells = self._reconstruct(parents_x, parents_y, start_cell, goal_cell)
                cells = self._smooth(cells, blocked)
                path = [grid.grid_to_world(cx, cy) for cx, cy in cells]
                return PlanResult(path=path, cost=cost * grid.resolution, expanded=expanded)
            for dx, dy, move_cost in self._moves:
                nx, ny = x + dx, y + dy
                if nx <= 0 or ny <= 0 or nx >= size - 1 or ny >= size - 1:
                    continue
                if blocked[ny, nx]:
                    continue
                if dx and dy and (blocked[y, nx] or blocked[ny, x]):
                    continue
                # Mildly prefer well-observed free space over cells near unknown.
                uncertainty = max(0.0, 1.0 + float(grid.log_odds[ny, nx])) * 0.08
                next_cost = cost + move_cost + uncertainty
                if next_cost + 1e-5 >= float(costs[ny, nx]):
                    continue
                costs[ny, nx] = next_cost
                parents_x[ny, nx], parents_y[ny, nx] = x, y
                heuristic = math.hypot(goal_cell[0] - nx, goal_cell[1] - ny)
                heapq.heappush(queue, (next_cost + heuristic, next_cost, nx, ny))
        return None

    @staticmethod
    def _reconstruct(
        parents_x: np.ndarray,
        parents_y: np.ndarray,
        start: GridCell,
        goal: GridCell,
    ) -> List[GridCell]:
        path = [goal]
        cursor = goal
        while cursor != start:
            x, y = cursor
            cursor = (int(parents_x[y, x]), int(parents_y[y, x]))
            if cursor[0] < 0:
                return []
            path.append(cursor)
        path.reverse()
        return path

    @staticmethod
    def _smooth(path: List[GridCell], blocked: np.ndarray) -> List[GridCell]:
        if len(path) < 3:
            return path
        result = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            furthest = anchor + 1
            for candidate in range(anchor + 2, len(path)):
                if OccupancyGrid.line_is_clear(blocked, path[anchor], path[candidate]):
                    furthest = candidate
                else:
                    break
            result.append(path[furthest])
            anchor = furthest
        return result


class FrontierExplorer:
    def __init__(self, config: PlannerConfig, planner: AStarPlanner):
        self.config = config
        self.planner = planner

    @staticmethod
    def _components(mask: np.ndarray) -> List[List[GridCell]]:
        visited = np.zeros_like(mask, dtype=bool)
        components: List[List[GridCell]] = []
        height, width = mask.shape
        for y, x in zip(*np.nonzero(mask)):
            if visited[y, x]:
                continue
            visited[y, x] = True
            queue = deque([(int(x), int(y))])
            component: List[GridCell] = []
            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
            components.append(component)
        return components

    def choose(self, grid: OccupancyGrid, pose: Pose2D) -> Tuple[Optional[WorldPoint], Optional[PlanResult]]:
        components = [
            component
            for component in self._components(grid.frontier_mask())
            if len(component) >= self.config.frontier_min_cells
        ]
        if not components:
            return None, None

        start_cell = grid.world_to_grid(pose.x, pose.y)
        candidates = []
        for component in components:
            centroid_x = sum(cell[0] for cell in component) / len(component)
            centroid_y = sum(cell[1] for cell in component) / len(component)
            representative = min(
                component,
                key=lambda cell: (cell[0] - centroid_x) ** 2 + (cell[1] - centroid_y) ** 2,
            )
            distance = math.hypot(representative[0] - start_cell[0], representative[1] - start_cell[1])
            optimistic = (
                self.config.information_gain_weight * len(component)
                - self.config.travel_cost_weight * distance * grid.resolution
            )
            candidates.append((optimistic, len(component), representative))
        candidates.sort(reverse=True)

        best = None
        for _, gain, cell in candidates[: self.config.frontier_candidate_limit]:
            goal = grid.grid_to_world(*cell)
            result = self.planner.plan(grid, (pose.x, pose.y), goal)
            if result is None or not result.path:
                continue
            score = self.config.information_gain_weight * gain - self.config.travel_cost_weight * result.cost
            if best is None or score > best[0]:
                best = (score, goal, result)
        if best is None:
            return None, None
        return best[1], best[2]


class DynamicWindowPlanner:
    """Acceleration-constrained critic-based local planner."""

    def __init__(
        self,
        robot: RobotConfig,
        config: PlannerConfig,
        dynamic_config: DynamicObstacleConfig,
    ):
        self.robot = robot
        self.config = config
        self.dynamic_config = dynamic_config

    @staticmethod
    def _lookahead(path: Sequence[WorldPoint], pose: Pose2D, distance: float) -> WorldPoint:
        if not path:
            return pose.x, pose.y
        for point in path:
            if math.hypot(point[0] - pose.x, point[1] - pose.y) >= distance:
                return point
        return path[-1]

    def command(
        self,
        pose: Pose2D,
        velocity: Velocity,
        path: Sequence[WorldPoint],
        scan: LaserScan,
        control_dt: float,
        dynamic_obstacles: Sequence[DynamicObstacle],
        now: float,
    ) -> ControlCommand:
        if not path:
            return ControlCommand(0.0, 0.55, "no-path search")
        control_dt = max(0.02, control_dt)
        v_low = max(0.0, velocity.linear - self.robot.max_linear_accel * control_dt)
        v_high = min(self.robot.max_linear_speed, velocity.linear + self.robot.max_linear_accel * control_dt)
        w_low = max(-self.robot.max_angular_speed, velocity.angular - self.robot.max_angular_accel * control_dt)
        w_high = min(self.robot.max_angular_speed, velocity.angular + self.robot.max_angular_accel * control_dt)
        v_samples = np.linspace(v_low, max(v_low, v_high), self.config.dwa_linear_samples)
        w_samples = np.linspace(w_low, max(w_low, w_high), self.config.dwa_angular_samples)
        obstacle_world = transform_points(scan.points_robot(include_max_range=False)[::2], pose)
        lookahead = self._lookahead(path, pose, self.config.lookahead_distance)
        goal = path[-1]

        best_score = -math.inf
        best_command: Optional[ControlCommand] = None
        for linear in v_samples:
            for angular in w_samples:
                score = self._score_trajectory(
                    pose,
                    float(linear),
                    float(angular),
                    obstacle_world,
                    dynamic_obstacles,
                    lookahead,
                    goal,
                    now,
                )
                if score > best_score:
                    best_score = score
                    best_command = ControlCommand(float(linear), float(angular), "DWA")

        if best_command is None:
            desired = math.atan2(lookahead[1] - pose.y, lookahead[0] - pose.x)
            turn = float(np.clip(1.4 * angle_difference(desired, pose.theta), -0.8, 0.8))
            return ControlCommand(0.0, turn if abs(turn) > 0.15 else 0.45, "DWA blocked")
        return best_command

    def _score_trajectory(
        self,
        pose: Pose2D,
        linear: float,
        angular: float,
        obstacles: np.ndarray,
        dynamic_obstacles: Sequence[DynamicObstacle],
        lookahead: WorldPoint,
        goal: WorldPoint,
        now: float,
    ) -> float:
        x, y, theta = pose.x, pose.y, pose.theta
        min_clearance = 5.0
        steps = max(1, int(self.config.dwa_horizon / self.config.dwa_dt))
        for _ in range(steps):
            theta += angular * self.config.dwa_dt
            x += linear * math.cos(theta) * self.config.dwa_dt
            y += linear * math.sin(theta) * self.config.dwa_dt
            if obstacles.size:
                clearance = float(np.min(np.hypot(obstacles[:, 0] - x, obstacles[:, 1] - y)))
                min_clearance = min(min_clearance, clearance)
                if clearance < self.robot.robot_radius + self.robot.safety_margin:
                    return -math.inf

        braking_distance = linear * linear / max(0.1, 2.0 * self.robot.max_linear_accel)
        if min_clearance < braking_distance + self.robot.robot_radius + self.robot.safety_margin:
            return -math.inf
        dynamic_prediction = predict_dynamic_clearance(
            pose,
            ControlCommand(linear, angular, "DWA candidate"),
            dynamic_obstacles,
            self.robot,
            self.dynamic_config,
            now,
            horizon=self.dynamic_config.planning_prediction_horizon,
            dt=self.config.dwa_dt,
        )
        if dynamic_prediction.ttc is not None:
            return -math.inf
        desired_heading = math.atan2(lookahead[1] - y, lookahead[0] - x)
        heading_score = math.cos(angle_difference(desired_heading, theta))
        old_goal_distance = math.hypot(goal[0] - pose.x, goal[1] - pose.y)
        new_goal_distance = math.hypot(goal[0] - x, goal[1] - y)
        progress = old_goal_distance - new_goal_distance
        path_distance = math.hypot(lookahead[0] - x, lookahead[1] - y)
        clearance_score = min(min_clearance, 1.0)
        dynamic_clearance_score = min(dynamic_prediction.min_clearance, 1.0)
        if not math.isfinite(dynamic_clearance_score):
            dynamic_clearance_score = 1.0
        return (
            3.2 * progress
            + 1.45 * heading_score
            + 1.05 * clearance_score
            + 1.20 * dynamic_clearance_score
            + 0.45 * linear / max(0.01, self.robot.max_linear_speed)
            - 0.65 * path_distance
            - 0.05 * abs(angular)
        )

