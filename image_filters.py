# image_filters.py
"""
Simple image filter manager.

Provides:
- A FrameFilter class
- Several basic filter modes: None, Grayscale, Blur, Edge
"""

import cv2


class FrameFilter:
    """
    Manage and apply simple image filters to frames.
    """

    VALID_MODES = ("None", "Grayscale", "Blur", "Edge")

    def __init__(self, mode: str = "None"):
        """
        :param mode: Initial filter mode.
        """
        self.mode = "None"
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Set the current filter mode (invalid values fallback to 'None')."""
        if mode not in self.VALID_MODES:
            mode = "None"
        self.mode = mode

    def apply(self, frame):
        """
        Apply the selected filter mode to the BGR frame.

        :param frame: Input BGR frame.
        :return: Processed BGR frame.
        """
        if self.mode == "Grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.mode == "Blur":
            return cv2.GaussianBlur(frame, (9, 9), 0)

        if self.mode == "Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # "None" or unknown → no extra processing
        return frame
