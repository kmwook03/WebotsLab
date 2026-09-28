"""On-robot occupancy grid diagnostics for engineering review."""

from typing import Optional, Sequence, Tuple

import numpy as np

from mapping import OccupancyGrid
from models import Pose2D


class MapDisplay:
    def __init__(self, display):
        self.display = display

    def draw(
        self,
        grid: OccupancyGrid,
        pose: Pose2D,
        path: Sequence[Tuple[float, float]],
        goal: Optional[Tuple[float, float]],
    ) -> None:
        if self.display is None:
            return
        width, height = self.display.getWidth(), self.display.getHeight()
        if width != grid.size or height != grid.size:
            return
        unknown = grid.unknown_mask()
        probability = grid.probability()
        gray = np.clip(255.0 * (1.0 - probability), 0, 255).astype(np.uint8)
        gray[unknown] = 128
        gray = np.flipud(gray)
        bgra = np.empty((height, width, 4), dtype=np.uint8)
        bgra[:, :, 0] = gray
        bgra[:, :, 1] = gray
        bgra[:, :, 2] = gray
        bgra[:, :, 3] = 255
        image = self.display.imageNew(bgra.tobytes(), self.display.BGRA, width, height)
        self.display.imagePaste(image, 0, 0, False)
        self.display.imageDelete(image)

        if len(path) > 1:
            self.display.setColor(0x00B7FF)
            cells = [grid.world_to_grid(*point) for point in path]
            for first, second in zip(cells[:-1], cells[1:]):
                self.display.drawLine(first[0], height - 1 - first[1], second[0], height - 1 - second[1])
        if goal is not None:
            gx, gy = grid.world_to_grid(*goal)
            self.display.setColor(0x00FF33)
            self.display.fillOval(gx, height - 1 - gy, 3, 3)
        rx, ry = grid.world_to_grid(pose.x, pose.y)
        self.display.setColor(0xFF3030)
        self.display.fillOval(rx, height - 1 - ry, 3, 3)

