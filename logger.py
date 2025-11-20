# logger.py
import csv
import time

class DetectionLogger:
    """
    DetectionLogger writes motion detection data to a CSV file.

    Each row includes:
    - Frame ID
    - Motion ID
    - Bounding box (x, y, w, h)
    - Timestamp
    - Drone mode
    - Safety state
    """

    def __init__(self, filepath):
        self.file = open(filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Write CSV header
        self.writer.writerow([
            "frame_id", "motion_id", "x", "y", "w", "h",
            "timestamp", "mode", "safe"
        ])

    def log(self, frame_id, boxes, drone):
        """Log all detection bounding boxes for the current frame."""
        now = time.time()

        if not boxes:
            # Log empty state if no detections
            self.writer.writerow([
                frame_id, 0, -1, -1, -1, -1, now, drone.mode, drone.safe
            ])
            return

        motion_id = 0
        for (x, y, w, h) in boxes:
            motion_id += 1
            self.writer.writerow([
                frame_id, motion_id, x, y, w, h,
                now, drone.mode, drone.safe
            ])

    def close(self):
        """Close the CSV file safely."""
        if not self.file.closed:
            self.file.close()
