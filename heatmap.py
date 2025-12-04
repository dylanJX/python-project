# heatmap.py
"""
Simple heatmap generator that accumulates animal positions
and renders a heatmap image.
"""

import numpy as np
import cv2
import os

class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.float32)

    def add_point(self, box):
        """Accumulate heat at animal center."""
        if box is None:
            return

        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2

        # clamp indices
        cx = min(max(cx, 0), self.width - 1)
        cy = min(max(cy, 0), self.height - 1)

        # increase heat
        self.grid[cy, cx] += 1

    def save_heatmap(self, path="heatmap.png"):
        """Convert grid to heatmap and save as PNG."""
        if not os.path.exists("results"):
            os.makedirs("results")

        heat = self.grid.copy()
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        heat = heat.astype("uint8")

        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join("results", path), heatmap)
        return os.path.join("results", path)
