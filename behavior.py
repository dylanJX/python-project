# behavior.py
"""
Simple wildlife behavior analysis based on bounding box movement.
Tracks:
- movement speed
- whether the animal is approaching a boundary area
- whether the animal is stationary
"""

import math

class BehaviorAnalyzer:
    def __init__(self):
        self.last_center = None

    def analyze(self, box, frame_width, frame_height):
        """
        Analyze movement and behavior characteristics.

        Args:
            box: (x, y, w, h)
            frame_width: video width
            frame_height: video height

        Returns:
            dict: {
                "movement": float,
                "is_near_border": bool,
                "status": str
            }
        """
        if box is None:
            self.last_center = None
            return {"movement": 0.0, "is_near_border": False, "status": "No animal"}

        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2

        # movement speed (distance between frame centers)
        if self.last_center is None:
            movement = 0.0
        else:
            px, py = self.last_center
            movement = math.dist((cx, cy), (px, py))

        self.last_center = (cx, cy)

        # behavior rule: near border?
        margin = 80
        near_border = (
            cx < margin or cx > frame_width - margin or
            cy < margin or cy > frame_height - margin
        )

        # movement classification
        if movement > 25:
            status = "Moving Fast"
        elif movement > 5:
            status = "Moving"
        else:
            status = "Stationary"

        if near_border:
            status += " (Near Border)"

        return {
            "movement": round(movement, 2),
            "is_near_border": near_border,
            "status": status
        }
