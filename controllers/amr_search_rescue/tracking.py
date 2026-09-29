"""Compact LiDAR-cluster tracking for moving-obstacle prediction."""

from dataclasses import dataclass, field
import math
from typing import List, Sequence, Tuple

import numpy as np

from config import DynamicObstacleConfig
from mapping import OccupancyGrid
from models import DynamicObstacle, DynamicObstacleFrame, LaserScan, Pose2D, transform_points


@dataclass
class _Observation:
    x: float
    y: float
    radius: float
    indices: np.ndarray


@dataclass
class _Track:
    track_id: int
    x: float
    y: float
    radius: float
    first_seen: float
    last_seen: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    moving_hits: int = 0
    history: List[Tuple[float, float, float]] = field(default_factory=list)


class DynamicObstacleTracker:
    def __init__(self, config: DynamicObstacleConfig):
        self.config = config
        self._tracks: List[_Track] = []
        self._next_id = 1

    def _clusters(
        self,
        pose: Pose2D,
        scan: LaserScan,
        grid: OccupancyGrid | None,
    ) -> List[_Observation]:
        valid = scan.valid_mask(include_max_range=False)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return []

        points_robot = np.column_stack(
            (
                scan.ranges[indices] * np.cos(scan.angles[indices]),
                scan.ranges[indices] * np.sin(scan.angles[indices]),
            )
        ).astype(np.float32)
        groups: List[List[int]] = [[0]]
        for offset in range(1, len(indices)):
            consecutive = int(indices[offset]) == int(indices[offset - 1]) + 1
            gap = float(np.linalg.norm(points_robot[offset] - points_robot[offset - 1]))
            if consecutive and gap <= self.config.cluster_gap:
                groups[-1].append(offset)
            else:
                groups.append([offset])

        # The scan starts and ends at the same rear bearing. Merge that cluster
        # when both ends are contiguous in Cartesian space.
        if len(groups) > 1 and indices[0] == 0 and indices[-1] == scan.ranges.size - 1:
            wrap_gap = float(np.linalg.norm(points_robot[groups[0][0]] - points_robot[groups[-1][-1]]))
            if wrap_gap <= self.config.cluster_gap:
                groups[0] = groups[-1] + groups[0]
                groups.pop()

        observations = []
        for group in groups:
            if len(group) < self.config.min_cluster_points:
                continue
            local = points_robot[group]
            extent = float(np.max(np.linalg.norm(local - np.mean(local, axis=0), axis=1)) * 2.0)
            if not self.config.min_cluster_width <= extent <= self.config.max_cluster_width:
                continue
            world = transform_points(local, pose)
            center = np.median(world, axis=0)
            radius = float(np.clip(0.5 * extent, self.config.obstacle_radius_min, self.config.obstacle_radius_max))
            direction = center - np.array([pose.x, pose.y], dtype=np.float32)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-5:
                center = center + radius * direction / direction_norm
            if grid is not None and grid.observed_count > 500:
                gx = np.rint(world[:, 0] / grid.resolution).astype(np.int32) + grid.origin
                gy = np.rint(world[:, 1] / grid.resolution).astype(np.int32) + grid.origin
                valid_grid = (gx >= 0) & (gx < grid.size) & (gy >= 0) & (gy < grid.size)
                if np.any(valid_grid):
                    static_ratio = float(np.mean(grid.log_odds[gy[valid_grid], gx[valid_grid]] > 1.8))
                    if static_ratio >= 0.60:
                        continue
            observations.append(
                _Observation(
                    float(center[0]),
                    float(center[1]),
                    radius,
                    indices[np.asarray(group, dtype=np.int32)],
                )
            )
        return observations

    def update(
        self,
        now: float,
        pose: Pose2D,
        scan: LaserScan,
        grid: OccupancyGrid | None = None,
    ) -> DynamicObstacleFrame:
        observations = self._clusters(pose, scan, grid)
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_observations = set(range(len(observations)))
        candidates = []
        for track_index, track in enumerate(self._tracks):
            elapsed = max(0.0, now - track.last_seen)
            predicted_x = track.x + track.vx * elapsed
            predicted_y = track.y + track.vy * elapsed
            gate = self.config.association_distance + min(0.25, math.hypot(track.vx, track.vy) * elapsed)
            for observation_index, observation in enumerate(observations):
                distance = math.hypot(observation.x - predicted_x, observation.y - predicted_y)
                if distance <= gate:
                    candidates.append((distance, track_index, observation_index))

        assignments = []
        for _, track_index, observation_index in sorted(candidates):
            if track_index not in unmatched_tracks or observation_index not in unmatched_observations:
                continue
            unmatched_tracks.remove(track_index)
            unmatched_observations.remove(observation_index)
            assignments.append((track_index, observation_index))

        observation_tracks = {}
        for track_index, observation_index in assignments:
            track = self._tracks[track_index]
            observation = observations[observation_index]
            track.history.append((now, observation.x, observation.y))
            track.history = [item for item in track.history if now - item[0] <= 0.90]
            if len(track.history) >= 2:
                history = np.asarray(track.history, dtype=np.float64)
                times = history[:, 0] - np.mean(history[:, 0])
                denominator = float(np.dot(times, times))
                if denominator > 1e-5:
                    raw_vx = float(np.dot(times, history[:, 1] - np.mean(history[:, 1])) / denominator)
                    raw_vy = float(np.dot(times, history[:, 2] - np.mean(history[:, 2])) / denominator)
                    raw_speed = math.hypot(raw_vx, raw_vy)
                    if raw_speed > 0.80:
                        scale = 0.80 / raw_speed
                        raw_vx *= scale
                        raw_vy *= scale
                    alpha = self.config.velocity_alpha
                    track.vx = (1.0 - alpha) * track.vx + alpha * raw_vx
                    track.vy = (1.0 - alpha) * track.vy + alpha * raw_vy
            speed = math.hypot(track.vx, track.vy)
            track.moving_hits = track.moving_hits + 1 if speed >= self.config.moving_speed else max(0, track.moving_hits - 1)
            track.x, track.y = observation.x, observation.y
            track.radius = 0.7 * track.radius + 0.3 * observation.radius
            track.last_seen = now
            track.hits += 1
            observation_tracks[observation_index] = track

        for observation_index in sorted(unmatched_observations):
            observation = observations[observation_index]
            track = _Track(
                self._next_id,
                observation.x,
                observation.y,
                observation.radius,
                now,
                now,
                history=[(now, observation.x, observation.y)],
            )
            self._next_id += 1
            self._tracks.append(track)
            observation_tracks[observation_index] = track

        self._tracks = [track for track in self._tracks if now - track.last_seen <= self.config.track_timeout]
        confirmed = []
        planning_tracks = []
        safety_tracks = []
        for track in self._tracks:
            provisional_confidence = min(1.0, 0.18 + 0.12 * track.hits)
            obstacle = DynamicObstacle(
                track.track_id,
                track.x,
                track.y,
                track.vx,
                track.vy,
                track.radius,
                provisional_confidence,
                track.last_seen,
            )
            if track.hits >= 2 or math.hypot(track.x - pose.x, track.y - pose.y) < 1.10:
                safety_tracks.append(obstacle)
            if (
                track.hits >= 2
                and track.moving_hits >= 1
                and math.hypot(track.vx, track.vy) >= self.config.moving_speed
            ):
                # Give DWA advance warning before the stricter confirmed-track
                # threshold is reached.  This is advisory planning input only;
                # stationary compact geometry is excluded by the motion test
                # and final actuation still passes through the safety guard.
                planning_tracks.append(obstacle)
            if track.hits < self.config.confirmation_hits or track.moving_hits < self.config.moving_confirmation_hits:
                continue
            confidence = min(1.0, 0.25 + 0.12 * track.hits)
            confirmed.append(
                DynamicObstacle(
                    track.track_id,
                    track.x,
                    track.y,
                    track.vx,
                    track.vy,
                    track.radius,
                    confidence,
                    track.last_seen,
                )
            )

        ignored = np.zeros(scan.ranges.size, dtype=bool)
        confirmed_ids = {track.track_id for track in confirmed}
        for observation_index, track in observation_tracks.items():
            if track.track_id in confirmed_ids:
                ignored[observations[observation_index].indices] = True
        planning_by_id = {obstacle.track_id: obstacle for obstacle in planning_tracks}
        # Confirmation is stronger than the provisional motion gate.  Keep
        # every confirmed track in DWA even if its filtered speed briefly dips
        # while the person turns or pauses.
        planning_by_id.update({obstacle.track_id: obstacle for obstacle in confirmed})
        planning_tracks = list(planning_by_id.values())
        return DynamicObstacleFrame(
            tuple(confirmed),
            tuple(planning_tracks),
            tuple(safety_tracks),
            ignored,
            len(self._tracks),
        )
