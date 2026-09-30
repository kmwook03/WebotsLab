"""Rule-based target perception using only RGB pixels and LiDAR ranges."""

import math
from typing import Optional, Tuple

import numpy as np

from config import TargetConfig, TraversabilityConfig
from models import LaserScan, TargetDetection, TraversabilityFrame


class TraversableAreaDetector:
    """Find bottom-connected floor pixels and align columns with LiDAR rays.

    The detector deliberately uses no simulator labels or learned model.  A
    robust colour reference is estimated from the image's bottom band, then
    each column is followed upward until a sustained non-floor region occurs.
    This makes an obstacle interrupt the visible floor even when similarly
    coloured floor remains visible on either side of it.
    """

    def __init__(self, config: Optional[TraversabilityConfig] = None):
        self.config = config or TraversabilityConfig()

    def _floor_mask(self, bgra: bytes, width: int, height: int) -> np.ndarray:
        image = np.frombuffer(bgra, dtype=np.uint8).reshape((height, width, 4))
        rgb = image[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        intensity = np.mean(rgb, axis=2)
        chromaticity = rgb / np.maximum(np.sum(rgb, axis=2, keepdims=True), 1e-4)

        band_height = max(2, int(round(height * self.config.reference_band_ratio)))
        reference_pixels = rgb[height - band_height :, :, :].reshape(-1, 3)
        reference_rgb = np.median(reference_pixels, axis=0)
        reference_intensity = float(np.mean(reference_rgb))
        reference_chromaticity = reference_rgb / max(float(np.sum(reference_rgb)), 1e-4)

        chromaticity_error = np.linalg.norm(
            chromaticity - reference_chromaticity[None, None, :], axis=2
        )
        intensity_error = np.abs(intensity - reference_intensity)
        raw_floor = (
            (chromaticity_error <= self.config.chromaticity_threshold)
            & (intensity_error <= self.config.intensity_threshold)
        )

        horizon = min(height - 1, max(0, int(round(height * self.config.horizon_ratio))))
        raw_floor[:horizon, :] = False
        connected = np.zeros_like(raw_floor)
        max_gap = max(0, self.config.max_vertical_gap)
        for x in range(width):
            misses = 0
            for y in range(height - 1, horizon - 1, -1):
                if raw_floor[y, x]:
                    misses = max(0, misses - 1)
                    connected[y, x] = True
                else:
                    misses += 1
                    if misses > max_gap:
                        break
        return connected

    def detect(
        self,
        bgra: bytes,
        width: int,
        height: int,
        field_of_view: float,
        scan: LaserScan,
    ) -> TraversabilityFrame:
        if not bgra or width <= 0 or height <= 0:
            return TraversabilityFrame.empty()

        floor_mask = self._floor_mask(bgra, width, height)
        horizon = min(height - 1, max(0, int(round(height * self.config.horizon_ratio))))
        stride = max(1, self.config.column_stride)
        bearings = []
        confidences = []
        lidar_ranges = []
        lidar_hits = []
        gate = math.radians(self.config.lidar_bearing_gate_deg)

        for start in range(0, width, stride):
            stop = min(width, start + stride)
            centre_x = 0.5 * (start + stop - 1)
            bearing = (0.5 - centre_x / max(1.0, width - 1.0)) * field_of_view
            confidence = float(np.mean(floor_mask[horizon:, start:stop]))
            difference = np.abs(
                np.arctan2(
                    np.sin(scan.angles - bearing),
                    np.cos(scan.angles - bearing),
                )
            )
            associated = difference <= gate
            hit_values = scan.ranges[associated & scan.valid_mask(include_max_range=False)]
            if hit_values.size:
                lidar_range = float(
                    np.percentile(hit_values, self.config.lidar_percentile)
                )
                lidar_hit = True
            else:
                lidar_range = float(scan.max_range)
                lidar_hit = False
            bearings.append(bearing)
            confidences.append(confidence)
            lidar_ranges.append(lidar_range)
            lidar_hits.append(lidar_hit)

        confidence_array = np.asarray(confidences, dtype=np.float32)
        if confidence_array.size >= 3:
            confidence_array = np.convolve(
                confidence_array,
                np.array([0.2, 0.6, 0.2], dtype=np.float32),
                mode="same",
            ).astype(np.float32)
        return TraversabilityFrame(
            bearings=np.asarray(bearings, dtype=np.float32),
            confidence=np.clip(confidence_array, 0.0, 1.0),
            lidar_ranges=np.asarray(lidar_ranges, dtype=np.float32),
            lidar_hits=np.asarray(lidar_hits, dtype=bool),
            floor_mask=floor_mask,
        )


class RedTargetDetector:
    """Detect a simple configured colour target without simulator labels."""

    def __init__(self, config: Optional[TargetConfig] = None):
        self.config = config or TargetConfig()
        self.filtered_confidence = 0.0
        self.filtered_bearing = 0.0

    @staticmethod
    def _largest_component(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best = None
        for y, x in zip(*np.nonzero(mask)):
            if visited[y, x]:
                continue
            stack = [(int(x), int(y))]
            visited[y, x] = True
            min_x = max_x = int(x)
            min_y = max_y = int(y)
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            candidate = (area, min_x, min_y, max_x, max_y)
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best

    def associate_range(self, scan: LaserScan, bearing: float) -> Optional[float]:
        difference = np.abs(np.arctan2(np.sin(scan.angles - bearing), np.cos(scan.angles - bearing)))
        mask = (
            (difference < math.radians(self.config.lidar_bearing_gate_deg))
            & scan.valid_mask(include_max_range=False)
        )
        values = scan.ranges[mask]
        if values.size == 0:
            return None
        return float(np.percentile(values, 30.0))

    def detect(
        self,
        bgra: bytes,
        width: int,
        height: int,
        field_of_view: float,
        scan: Optional[LaserScan] = None,
    ) -> TargetDetection:
        if not bgra:
            return TargetDetection()
        image = np.frombuffer(bgra, dtype=np.uint8).reshape((height, width, 4))
        # Half resolution is enough for a coloured rescue marker and bounds CPU.
        sample = image[::2, ::2, :]
        primary = sample[:, :, self.config.primary_channel].astype(np.int16)
        secondary_a = sample[:, :, self.config.secondary_channels[0]].astype(np.int16)
        secondary_b = sample[:, :, self.config.secondary_channels[1]].astype(np.int16)
        mask = (
            (primary > self.config.min_primary)
            & (primary > secondary_a * self.config.primary_ratio)
            & (primary > secondary_b * self.config.primary_ratio)
            & ((primary - np.maximum(secondary_a, secondary_b)) > self.config.primary_margin)
        )

        # A 3-neighbour filter rejects isolated hot pixels and texture noise.
        neighbours = mask.astype(np.uint8)
        neighbours[1:, :] += mask[:-1, :]
        neighbours[:-1, :] += mask[1:, :]
        neighbours[:, 1:] += mask[:, :-1]
        neighbours[:, :-1] += mask[:, 1:]
        clean = mask & (neighbours >= 3)
        component = self._largest_component(clean)
        if component is None or component[0] < self.config.min_component_area:
            self.filtered_confidence *= 0.72
            return TargetDetection(confidence=self.filtered_confidence)

        area, min_x, min_y, max_x, max_y = component
        box_width, box_height = max_x - min_x + 1, max_y - min_y + 1
        area_ratio = area / float(clean.size)
        fill_ratio = area / float(box_width * box_height)
        raw_confidence = min(1.0, area / 60.0) * min(1.0, fill_ratio / 0.45)
        centre_x = 0.5 * (min_x + max_x)
        bearing = (0.5 - centre_x / clean.shape[1]) * field_of_view
        self.filtered_confidence = 0.58 * self.filtered_confidence + 0.42 * raw_confidence
        self.filtered_bearing = 0.62 * self.filtered_bearing + 0.38 * bearing
        # The target's visual dimensions are supplied by the challenge. Keep a
        # monocular estimate even when a LiDAR return exists so an unrelated
        # obstacle near the same bearing cannot cause a premature rescue stop.
        vertical_fov = 2.0 * math.atan(math.tan(0.5 * field_of_view) * height / width)
        focal_y = 0.5 * height / max(1e-6, math.tan(0.5 * vertical_fov))
        full_resolution_height = max(1.0, 2.0 * box_height)
        monocular_distance = float(
            np.clip(self.config.known_height_m * focal_y / full_resolution_height, 0.30, 8.0)
        )
        lidar_distance = self.associate_range(scan, self.filtered_bearing) if scan is not None else None
        gate = max(
            self.config.lidar_range_gate_min,
            self.config.lidar_range_gate_ratio * monocular_distance,
        )
        if lidar_distance is not None and abs(lidar_distance - monocular_distance) <= gate:
            distance = 0.72 * lidar_distance + 0.28 * monocular_distance
        else:
            distance = monocular_distance
        return TargetDetection(
            seen=self.filtered_confidence >= self.config.seen_confidence,
            bearing=self.filtered_bearing,
            confidence=self.filtered_confidence,
            area_ratio=area_ratio,
            range_m=distance,
            bbox=(2 * min_x, 2 * min_y, 2 * max_x, 2 * max_y),
        )

