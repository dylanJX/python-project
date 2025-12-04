# detector.py
"""
Wildlife-like motion detector using background subtraction.

This module provides a WildlifeDetector class that:
- Maintains an internal background model (MOG2)
- Detects moving regions as potential "wildlife activity"
- Returns bounding boxes for detected objects
"""

import cv2
#import numpy as np


class WildlifeDetector:
    """
    Detect moving objects (e.g., animals) using background subtraction.
    """

    def __init__(self, history: int = 500, var_threshold: int = 40,
                 min_area: int = 1500, detect_shadows: bool = True):
        """
        :param history: How many frames the background model remembers.
        :param var_threshold: Sensitivity threshold (lower is more sensitive).
        :param min_area: Minimum contour area to be considered a detection.
        :param detect_shadows: Whether to let MOG2 model shadows.
        """
        self.history = history
        self.var_threshold = var_threshold
        self.min_area = min_area

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=detect_shadows
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def set_min_area(self, min_area: int) -> None:
        """Update the minimum contour area."""
        self.min_area = max(0, int(min_area))

    def set_sensitivity(self, var_threshold: int) -> None:
        """
        Update the sensitivity (varThreshold).
        Recreate the background subtractor with the new parameter.
        """
        self.var_threshold = max(1, int(var_threshold))
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )

    def reset_model(self) -> None:
        """Reset the background model."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )

    def detect(self, frame):
        """
        Run motion detection on a single BGR frame.

        :param frame: Input BGR frame (numpy array).
        :return: List of bounding boxes [(x, y, w, h), ...]
        """
        # Apply background subtraction
        fgmask = self._subtractor.apply(frame)

        # Remove shadows (in MOG2 shadows are usually ~127)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean noise
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, self._kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

        return boxes
