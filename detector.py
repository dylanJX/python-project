# detector.py
"""
YOLO-based object detector.

Supports:
- Raw YOLO detection (detect_raw)
- Filtered detection for teammate to modify (detect_filtered)
- Legacy detect() function for backward compatibility
"""

from typing import List, Dict, Tuple, Optional
import cv2
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]


class WildlifeDetector:
    """
    YOLO-based object detector with support for raw and filtered pipelines.
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

    # -------------------------------------------------------------
    # Shared YOLO → detection dictionary parser
    # -------------------------------------------------------------
    def _parse_yolo_output(self, result):
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        names = result.names

        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < self.conf_threshold:
                continue
            if self.classes is not None and cls_id not in self.classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            # Integrate your min_area filtering logic
            if self.min_area > 0 and (w * h) < self.min_area:
                continue

            detections.append(
                {
                    "box": (x1, y1, w, h),
                    "label": names.get(cls_id, f"id_{cls_id}"),
                    "conf": conf,
                }
            )

        return detections

    # -------------------------------------------------------------
    # RAW YOLO detection
    # -------------------------------------------------------------
    def detect_raw(self, frame) -> List[Dict]:
        """
        Pure YOLO detection.
        """
        results = self.model(frame, verbose=False)
        if not results:
            return []
        return self._parse_yolo_output(results[0])

    def detect_filtered(self, frame) -> List[Dict]:
        """
        Run YOLO first, then allow custom filtering logic.
        Currently same as raw version.
        """

        detections = self.detect_raw(frame)

        return detections

    # -------------------------------------------------------------
    # Legacy detect() = alias for raw
    # -------------------------------------------------------------
    def detect(self, frame) -> List[Dict]:
        """Backward compatibility: same as raw YOLO."""
        return self.detect_raw(frame)

    # -------------------------------------------------------------
    def set_min_area(self, value: int) -> None:
        self.min_area = int(value)
