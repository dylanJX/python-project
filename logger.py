# logger.py
"""
CSV logger for wildlife detections and statistics.

Each row may contain:
- Frame index
- Number of detections
- FPS
- Elapsed time
"""

import csv
import os
from typing import Optional


class DetectionLogger:
    """
    Append detection and performance statistics to a CSV file.
    """

    def __init__(self, filename: str = "wildlife_log.csv"):
        """
        :param filename: Path to the CSV file.
        """
        self.filename = filename
        self._file = None
        self._writer: Optional[csv.writer] = None

        # If the file does not exist, we will write a header.
        self._need_header = not os.path.exists(self.filename)

        # Open in append mode
        self._file = open(self.filename, mode="a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        if self._need_header:
            self._writer.writerow(["frame", "detections", "fps", "elapsed_seconds"])

    def log(self, frame_index: int, detections: int, fps: float, elapsed: float) -> None:
        """Append a single row to the CSV file."""
        if self._writer is None:
            return
        self._writer.writerow([frame_index, detections, f"{fps:.3f}", f"{elapsed:.3f}"])

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
