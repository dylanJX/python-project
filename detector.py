# detector.py
"""
YOLO-based object detector.

Replaces the old background-subtraction WildlifeDetector with a detector
that uses a YOLO model to find objects in each frame.

The detector returns a list of detection dictionaries, each with:
    - box: (x, y, w, h)  integer pixel coordinates
    - label: str         class name (e.g. "bird", "person")
    - conf: float        confidence between 0 and 1
"""

from typing import List, Dict, Tuple, Optional

import cv2
from ultralytics import YOLO


BBox = Tuple[int, int, int, int]


class WildlifeDetector:
    """
    YOLO-based object detector.

    Parameters
    ----------
    model_path : str
        Path to a YOLO model file, e.g. "yolov8n.pt".
    conf_threshold : float
        Minimum confidence required to keep a detection.
    classes : Optional[list[int]]
        Optional list of class IDs to keep. If None, all classes are allowed.
    min_area : int
        Optional area filter in pixels^2. Detections with box area smaller
        than this value are discarded.
    """

    def __init__(
        self,
        min_area: int = 0,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        classes: Optional[list] = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.min_area = int(min_area) if min_area is not None else 0

    # ------------------------------------------------------------------
    def detect(self, frame) -> List[Dict]:
        """
        Run YOLO on the given BGR frame and return a list of detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by OpenCV.

        Returns
        -------
        list of dict
            Each dict has keys:
                - "box": (x, y, w, h)
                - "label": str
                - "conf": float
        """
        detections: List[Dict] = []

        # Ultralytics YOLO accepts BGR directly.
        results = self.model(frame, verbose=False)
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        names = result.names  # mapping from class id to string label

        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < self.conf_threshold:
                continue
            if self.classes is not None and cls_id not in self.classes:
                continue

            # xyxy format: (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)

            w = max(0, x2_i - x1_i)
            h = max(0, y2_i - y1_i)

            # Optional area filter (keeps compatibility with GUI slider)
            area = w * h
            if self.min_area > 0 and area < self.min_area:
                continue

            label = names.get(cls_id, f"id_{cls_id}")

            detections.append(
                {
                    "box": (x1_i, y1_i, w, h),
                    "label": label,
                    "conf": conf,
                }
            )

        return detections

    # ------------------------------------------------------------------
    def set_min_area(self, value: int) -> None:
        """Update the minimum area threshold."""
        self.min_area = int(value)
