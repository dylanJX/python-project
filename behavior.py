# behavior.py
"""
Behavior analysis based on object motion.

For each track (one object ID), we classify motion as:
- Stationary
- Moving Slowly
- Moving Fast

This version:
- Works per-object (uses track["id"])
- Uses Kalman velocity if available (vx, vy from kf_state)
- Normalizes speed by *object size* (box height/width), not full frame
- Still provides a global-normalized speed
- Smooths speed with an exponential moving average (EMA)
"""

import math
from typing import Dict, Tuple


class BehaviorAnalyzer:
    def __init__(self, smoothing: float = 0.7):

        self.speed_ema: Dict[int, float] = {}
        self.smoothing = float(smoothing)

    # ------------------------------------------------------------------
    def analyze(self, track: Dict, frame_width: int, frame_height: int) -> Dict:

        obj_id = int(track["id"])
        box = track["box"]
        center = track["center"]

        #  Estimate instantaneous speed (pixels per frame)
        speed = self._estimate_speed(track)

        #  Smooth with EMA to avoid flicker
        prev = self.speed_ema.get(obj_id, None)
        if prev is None:
            smoothed_speed = speed
        else:
            alpha = self.smoothing
            smoothed_speed = alpha * prev + (1.0 - alpha) * speed
        self.speed_ema[obj_id] = smoothed_speed

        #  Normalize by object size (relative motion)
        x, y, w, h = box
        obj_scale = max(1.0, float(max(w, h)))  # avoid division by zero
        speed_rel = smoothed_speed / obj_scale

        # Also keep the old global normalization if you want to inspect it
        diag = math.hypot(frame_width, frame_height)
        speed_norm = smoothed_speed / diag if diag > 0 else 0.0

        #  Classify motion using *relative* speed
        # Tuned for your horse / bird:
        #   < 0.01  -> Stationary (movement < 1% of object size per frame)
        #   < 0.05  -> Moving Slowly
        #   else    -> Moving Fast
        if speed_rel < 0.01:
            status = "Stationary"
        elif speed_rel < 0.05:
            status = "Moving Slowly"
        else:
            status = "Moving Fast"

        #  Border check (10% margin)
        is_near_border = self._is_near_border(
            box, frame_width, frame_height, margin_ratio=0.10
        )
        if is_near_border:
            status += " (Near Border)"

        return {
            "speed_px": round(smoothed_speed, 2),
            "speed_rel": round(speed_rel, 4),
            "speed_norm": round(speed_norm, 4),
            "is_near_border": is_near_border,
            "status": status,
        }

    # ------------------------------------------------------------------
    def _estimate_speed(self, track: Dict) -> float:

        kf_state = track.get("kf_state", None)
        if kf_state is not None and kf_state.shape[0] >= 4:
            vx = float(kf_state[2, 0])
            vy = float(kf_state[3, 0])
            return math.hypot(vx, vy)

        cx, cy = track["center"]
        last_center = track.get("last_center", None)
        if last_center is None:
            return 0.0
        px, py = last_center
        return math.hypot(cx - px, cy - py)

    # ------------------------------------------------------------------
    def _is_near_border(
        self,
        box: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        margin_ratio: float = 0.10,
    ) -> bool:
        """
        Check whether the box is near the image border (within a percentage margin).
        """
        x, y, w, h = box
        margin_x = int(frame_width * margin_ratio)
        margin_y = int(frame_height * margin_ratio)

        left = x
        top = y
        right = x + w
        bottom = y + h

        near_left = left <= margin_x
        near_top = top <= margin_y
        near_right = right >= frame_width - margin_x
        near_bottom = bottom >= frame_height - margin_y

        return near_left or near_top or near_right or near_bottom
