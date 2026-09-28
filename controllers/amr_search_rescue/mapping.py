"""Log-odds occupancy grid mapping with footprint inflation."""

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np

from config import MapConfig
from models import LaserScan, Pose2D


GridCell = Tuple[int, int]


def bresenham(start: GridCell, end: GridCell) -> Iterable[GridCell]:
    """Yield integer grid cells on a line, including both endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy


class OccupancyGrid:
    """Fixed-size robot-start-centred occupancy grid.

    Unknown is exactly zero log-odds. Negative evidence is free; positive
    evidence is occupied. The map's world frame is the supplied initial pose,
    never the simulator's ground-truth frame.
    """

    def __init__(self, config: MapConfig):
        self.config = config
        self.size = config.size
        self.resolution = config.resolution
        self.origin = self.size // 2
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)
        self.last_hit = np.full((self.size, self.size), -1, dtype=np.int32)
        self.update_count = 0
        self._inflation_cache = {}

    def in_bounds(self, cell: GridCell, margin: int = 0) -> bool:
        x, y = cell
        return margin <= x < self.size - margin and margin <= y < self.size - margin

    def world_to_grid(self, x: float, y: float) -> GridCell:
        return (
            int(round(x / self.resolution)) + self.origin,
            int(round(y / self.resolution)) + self.origin,
        )

    def grid_to_world(self, x: int, y: int) -> Tuple[float, float]:
        return ((x - self.origin) * self.resolution, (y - self.origin) * self.resolution)

    def update(self, pose: Pose2D, scan: LaserScan) -> None:
        self.update_count += 1
        origin = self.world_to_grid(pose.x, pose.y)
        if not self.in_bounds(origin, 1):
            return

        indices = range(0, scan.ranges.size, self.config.ray_stride)
        for index in indices:
            measured = float(scan.ranges[index])
            if not math.isfinite(measured) or measured < scan.min_range:
                continue
            hit = measured < scan.max_range * 0.985
            distance = min(measured, scan.max_range)
            angle = pose.theta + float(scan.angles[index])
            endpoint = self.world_to_grid(
                pose.x + distance * math.cos(angle),
                pose.y + distance * math.sin(angle),
            )
            cells = list(bresenham(origin, endpoint))
            if len(cells) < 2:
                continue
            free_cells = cells[1:-1] if hit else cells[1:]
            for cell in free_cells:
                if not self.in_bounds(cell, 1):
                    break
                x, y = cell
                self.log_odds[y, x] += self.config.free_log_odds
            if hit and self.in_bounds(cells[-1], 1):
                x, y = cells[-1]
                self.log_odds[y, x] += self.config.occupied_log_odds
                self.last_hit[y, x] = self.update_count

        np.clip(
            self.log_odds,
            self.config.min_log_odds,
            self.config.max_log_odds,
            out=self.log_odds,
        )
        if self.update_count % 30 == 0:
            self._decay_stale_obstacles()
        self._inflation_cache.clear()

    def _decay_stale_obstacles(self) -> None:
        age = self.update_count - self.last_hit
        stale = (self.log_odds > 0.0) & (age > self.config.stale_hit_steps)
        self.log_odds[stale] *= self.config.stale_decay

    @property
    def observed_count(self) -> int:
        return int(np.count_nonzero(np.abs(self.log_odds) > 0.05))

    def probability(self) -> np.ndarray:
        clipped = np.clip(self.log_odds, -10.0, 10.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def occupied_mask(self) -> np.ndarray:
        return self.log_odds >= self.config.occupied_threshold

    def free_mask(self) -> np.ndarray:
        return self.log_odds <= self.config.free_threshold

    def unknown_mask(self) -> np.ndarray:
        return np.abs(self.log_odds) < 0.05

    def inflated_obstacles(self, radius_m: float) -> np.ndarray:
        radius_cells = max(1, int(math.ceil(radius_m / self.resolution)))
        cached = self._inflation_cache.get(radius_cells)
        if cached is not None:
            return cached.copy()
        occupied = self.occupied_mask()
        inflated = occupied.copy()
        offsets: List[Tuple[int, int]] = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    offsets.append((dx, dy))
        ys, xs = np.nonzero(occupied)
        for dx, dy in offsets:
            shifted_x = xs + dx
            shifted_y = ys + dy
            valid = (
                (shifted_x >= 0)
                & (shifted_x < self.size)
                & (shifted_y >= 0)
                & (shifted_y < self.size)
            )
            inflated[shifted_y[valid], shifted_x[valid]] = True
        self._inflation_cache[radius_cells] = inflated.copy()
        return inflated

    def frontier_mask(self) -> np.ndarray:
        """Known-free cells with at least one 4-connected unknown neighbour."""
        free = self.free_mask()
        unknown = self.unknown_mask()
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        frontier = free & adjacent_unknown
        frontier[[0, -1], :] = False
        frontier[:, [0, -1]] = False
        return frontier

    def nearest_free(self, cell: GridCell, max_radius: int = 18) -> Optional[GridCell]:
        if self.in_bounds(cell) and self.free_mask()[cell[1], cell[0]]:
            return cell
        free = self.free_mask()
        cx, cy = cell
        for radius in range(1, max_radius + 1):
            candidates = []
            for dy in range(-radius, radius + 1):
                for dx in (-radius, radius):
                    candidates.append((cx + dx, cy + dy))
            for dx in range(-radius + 1, radius):
                for dy in (-radius, radius):
                    candidates.append((cx + dx, cy + dy))
            for candidate in candidates:
                if self.in_bounds(candidate) and free[candidate[1], candidate[0]]:
                    return candidate
        return None

    @staticmethod
    def line_is_clear(blocked: np.ndarray, start: GridCell, end: GridCell) -> bool:
        height, width = blocked.shape
        for x, y in bresenham(start, end):
            if x < 0 or y < 0 or x >= width or y >= height or blocked[y, x]:
                return False
        return True

