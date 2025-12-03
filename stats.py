# stats.py
"""
Statistics helper module.

Tracks:
- Total number of processed frames
- Elapsed time (seconds)
- Approximate FPS
"""

import time


class StatsTracker:
    """
    Track frame count, elapsed time and FPS for the video processing loop.
    """

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.total_frames = 0
        self.fps = 0.0

    def update(self) -> None:
        """Call this once per processed frame to update counters."""
        self.total_frames += 1
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now

    @property
    def elapsed(self) -> float:
        """Return total elapsed time in seconds since the tracker was created."""
        return time.time() - self.start_time
