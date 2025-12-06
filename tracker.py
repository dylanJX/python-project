# tracker.py
"""
Kalman-based multi-object tracker with IoU-first matching.

Each track uses a 2D constant-velocity Kalman filter:
    state = [x, y, vx, vy]^T
where (x, y) is the object center in pixels.

https://github.com/abewley/sort
https://github.com/zziz/kalman-filter
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math
import os
import csv

import numpy as np

BBox = Tuple[int, int, int, int]


class ObjectTracker:
    """
    Multi-object tracker using per-object Kalman filters and IoU-based matching.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        max_distance: float = 120.0,
        max_missed: int = 15,
        min_iou: float = 0.15,
    ) -> None:
        """
        Parameters
        ----------
        frame_width : int
            Frame width in pixels.
        frame_height : int
            Frame height in pixels.
        max_distance : float
            Maximum allowed distance (in pixels) between a predicted track center
            and a detection center to consider them a match (used as fallback
            when IoU is small).
        max_missed : int
            Maximum number of consecutive frames a track may be unmatched before
            it is removed.
        min_iou : float
            Minimum IoU required to accept a match; if IoU is lower but the
            distance is still small enough, distance is used as a fallback.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.min_iou = float(min_iou)

        # Internal track storage
        self._tracks: Dict[int, Dict] = {}
        self._finished_tracks: List[Dict] = []
        self._next_id: int = 1

        # Kalman filter matrices (same for all tracks)
        self._dt = 1.0  # assume ~1 frame step; can be tuned

        # State transition (constant velocity model)
        self._F = np.array(
            [
                [1.0, 0.0, self._dt, 0.0],
                [0.0, 1.0, 0.0, self._dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Measurement matrix: we observe x, y only
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Process noise covariance (Q) and measurement noise (R)
        q = 1e-2
        r = 5.0
        self._Q = q * np.eye(4, dtype=np.float32)
        self._R = r * np.eye(2, dtype=np.float32)
        self._I = np.eye(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, boxes: List[BBox], frame_index: int) -> List[Dict]:

        # No existing tracks: start a new track per box
        if not self._tracks:
            for box in boxes:
                self._start_new_track(box, frame_index)
            return [
                {
                    "id": t["id"],
                    "box": t["box"],
                    "center": t["center"],
                }
                for t in self._tracks.values()
            ]

        # 1) Kalman predict step for all tracks
        for track in self._tracks.values():
            self._kalman_predict(track)

        # 2) Assign detections to tracks using IoU + distance
        assignments: Dict[int, int] = {}  # det_index -> track_id
        unmatched_track_ids = set(self._tracks.keys())
        detection_centers = [self._center_of(b) for b in boxes]

        for det_index, (box, meas_center) in enumerate(
            zip(boxes, detection_centers)
        ):
            best_id: Optional[int] = None
            best_iou: float = 0.0
            best_dist: float = float("inf")

            for track_id in unmatched_track_ids:
                track = self._tracks[track_id]

                pred_box = track["pred_box"]
                iou = self._iou(box, pred_box)

                pcx, pcy = track["pred_center"]
                mcx, mcy = meas_center
                dist = math.hypot(mcx - pcx, mcy - pcy)

                # Prefer higher IoU; if IoU ties, prefer smaller distance
                if iou > best_iou or (iou == best_iou and dist < best_dist):
                    best_iou = iou
                    best_dist = dist
                    best_id = track_id

            # Accept match if IoU high enough, or distance small enough
            if best_id is not None and (
                best_iou >= self.min_iou or best_dist <= self.max_distance
            ):
                assignments[det_index] = best_id
                unmatched_track_ids.remove(best_id)

        matched_track_ids = set(assignments.values())

        # 3) Update matched tracks with measurements
        active_track_ids = set()
        for det_index, track_id in assignments.items():
            meas_box = boxes[det_index]
            meas_center = detection_centers[det_index]
            track = self._tracks[track_id]

            # Update Kalman state with measured center
            self._kalman_update(track, meas_center)

            # Update bookkeeping
            track["box"] = meas_box
            track["center"] = track["kf_state"][0, 0], track["kf_state"][1, 0]
            track["last_frame"] = frame_index

            # Path length (between previous center and new one)
            last_cx, last_cy = track["last_center"]
            cx, cy = track["center"]
            step_dist = math.hypot(cx - last_cx, cy - last_cy)
            track["path_length"] += step_dist
            track["last_center"] = (cx, cy)

            track["missed"] = 0
            track["history"].append((frame_index, meas_box))

            active_track_ids.add(track_id)

        # 4) Age unmatched tracks (no measurement this frame)
        to_finish = []
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_track_ids:
                track = self._tracks[track_id]
                track["missed"] += 1

                # Use predicted center/box as the current state
                track["center"] = track["pred_center"]
                track["box"] = track["pred_box"]
                track["last_frame"] = frame_index

                # Optional: accumulate path length based on prediction
                last_cx, last_cy = track["last_center"]
                cx, cy = track["center"]
                step_dist = math.hypot(cx - last_cx, cy - last_cy)
                track["path_length"] += step_dist
                track["last_center"] = (cx, cy)

                if track["missed"] > self.max_missed:
                    to_finish.append(track_id)

        for track_id in to_finish:
            self._finished_tracks.append(self._tracks.pop(track_id))

        # 5) Any detection not assigned → new track
        assigned_det_indices = set(assignments.keys())
        for det_index, box in enumerate(boxes):
            if det_index in assigned_det_indices:
                continue
            self._start_new_track(box, frame_index)
            new_id = self._next_id - 1
            active_track_ids.add(new_id)

        # Build list of active tracks for this frame (those with a detection now)
        # Build list of active tracks for this frame (those with a detection now)
        active_tracks = []
        for track_id in active_track_ids:
            t = self._tracks.get(track_id)
            if t is None:
                continue

            # Return a rich view of the track so behavior analysis
            # can use velocity, last_center, etc.
            active_tracks.append(
                {
                    "id": t["id"],
                    "box": t["box"],
                    "center": t["center"],
                    "last_center": t.get("last_center"),
                    "kf_state": t.get("kf_state"),
                    "path_length": t.get("path_length", 0.0),
                    "first_frame": t.get("first_frame"),
                    "last_frame": t.get("last_frame"),
                }
            )

        return active_tracks

    # ------------------------------------------------------------------
    # Summaries / export
    # ------------------------------------------------------------------
    def get_all_tracks(self) -> List[Dict]:
        """Return all tracks (finished + still active)."""
        return self._finished_tracks + list(self._tracks.values())

    def get_object_summaries(self) -> List[Dict]:
        """
        Build a compact summary for each track.

        For each track, we compute:
            - id
            - dwell_frames
            - path_length
            - avg_speed_per_frame
        """
        summaries: List[Dict] = []
        for tr in self.get_all_tracks():
            first = tr.get("first_frame", 1)
            last = tr.get("last_frame", first)
            dwell_frames = max(1, last - first + 1)
            path_length = tr.get("path_length", 0.0)

            if dwell_frames > 1:
                avg_speed = path_length / (dwell_frames - 1)
            else:
                avg_speed = 0.0

            summaries.append(
                {
                    "id": tr["id"],
                    "dwell_frames": dwell_frames,
                    "path_length": path_length,
                    "avg_speed_per_frame": avg_speed,
                }
            )
        return summaries

    def export_object_summaries_csv(self, path: str, fps: Optional[float] = None) -> str:
        """
        Export per-object statistics to a CSV file.

        If fps is provided, also includes approximate dwell time in seconds
        and average speed in pixels/second.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        summaries = self.get_object_summaries()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["id", "dwell_frames", "path_length_px", "avg_speed_px_per_frame"]
            if fps is not None and fps > 0:
                header.extend(["dwell_seconds", "avg_speed_px_per_second"])
            writer.writerow(header)

            for s in summaries:
                row = [
                    s["id"],
                    s["dwell_frames"],
                    f"{s['path_length']:.3f}",
                    f"{s['avg_speed_per_frame']:.3f}",
                ]
                if fps is not None and fps > 0:
                    dwell_sec = s["dwell_frames"] / fps
                    avg_speed_sec = s["avg_speed_per_frame"] * fps
                    row.extend([f"{dwell_sec:.3f}", f"{avg_speed_sec:.3f}"])
                writer.writerow(row)

        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_new_track(self, box: BBox, frame_index: int) -> None:
        """Create a new track initialized from a detection box."""
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Initial state: position = measured center, velocity = 0
        state = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        P = np.eye(4, dtype=np.float32) * 10.0  # fairly uncertain

        track = {
            "id": self._next_id,
            "box": box,
            "center": (cx, cy),
            "last_center": (cx, cy),
            "kf_state": state,
            "kf_P": P,
            "missed": 0,
            "first_frame": frame_index,
            "last_frame": frame_index,
            "history": [(frame_index, box)],
            "path_length": 0.0,
            # Will be filled in by predict
            "pred_center": (cx, cy),
            "pred_box": box,
        }
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _kalman_predict(self, track: Dict) -> None:
        """Kalman filter prediction step for a single track."""
        state = track["kf_state"]
        P = track["kf_P"]

        # x_k|k-1 = F x_k-1|k-1
        state_pred = self._F @ state
        P_pred = self._F @ P @ self._F.T + self._Q

        track["kf_state"] = state_pred
        track["kf_P"] = P_pred

        cx = float(state_pred[0, 0])
        cy = float(state_pred[1, 0])
        track["pred_center"] = (cx, cy)

        # Build a predicted box using the last known box size
        x, y, w, h = track["box"]
        px = int(cx - w / 2.0)
        py = int(cy - h / 2.0)

        # Clamp to frame bounds (optional but safe)
        px = max(0, min(self.frame_width - w, px))
        py = max(0, min(self.frame_height - h, py))

        track["pred_box"] = (px, py, w, h)

    def _kalman_update(self, track: Dict, meas_center: Tuple[float, float]) -> None:
        """Kalman filter update step for a matched track."""
        state_pred = track["kf_state"]
        P_pred = track["kf_P"]

        z = np.array([[meas_center[0]], [meas_center[1]]], dtype=np.float32)

        # y = z - H x
        y = z - (self._H @ state_pred)
        S = self._H @ P_pred @ self._H.T + self._R
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        # x_k|k = x_k|k-1 + K y
        state_upd = state_pred + K @ y
        P_upd = (self._I - K @ self._H) @ P_pred

        track["kf_state"] = state_upd
        track["kf_P"] = P_upd

    def _center_of(self, box: BBox) -> Tuple[float, float]:
        """Compute center of (x, y, w, h)."""
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    def _iou(self, box_a: BBox, box_b: BBox) -> float:
        """Intersection-over-Union between two boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / float(denom)
