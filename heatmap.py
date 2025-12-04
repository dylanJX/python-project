# heatmap.py
import os
import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # accumulation map
        self.map = np.zeros((height, width), dtype=np.float32)

    def add_point(self, bbox):
        """
        Add one detection to the heatmap.
        bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        # clamp to valid range
        cx = max(0, min(self.width - 1, cx))
        cy = max(0, min(self.height - 1, cy))

        self.map[cy, cx] += 1.0

    def save_heatmap(self, out_path="heatmap/heatmap.png"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if self.map.max() <= 0:
            # no data collected -> generate empty heatmap
            base = np.zeros_like(self.map, dtype=np.uint8)
        else:
            # optional blur to spread the points (looks better)
            blurred = cv2.GaussianBlur(self.map, (0, 0), sigmaX=15, sigmaY=15)

            # normalize to 0–255
            norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
            base = norm.astype("uint8")

        heat = cv2.applyColorMap(base, cv2.COLORMAP_JET)
        cv2.imwrite(out_path, heat)
        return out_path
