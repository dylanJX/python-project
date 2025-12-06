# gui_app.py
"""
Graphical application for wildlife-like activity analysis using YOLO.

This app:
- opens a camera or video file,
- runs YOLO-based object detection,
- tracks objects over time with persistent IDs,
- shows FPS and detection statistics,
- allows basic image filters,
- lets the user record video, save snapshots,
  export a heatmap and per-object statistics.
"""

import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

from PIL import Image, ImageTk

from detector import WildlifeDetector
from image_filters import FrameFilter
from stats import StatsTracker
from logger import DetectionLogger
from behavior import BehaviorAnalyzer
from heatmap import HeatmapGenerator
from tracker import ObjectTracker


class VisionVideoApp:
    """
    Main GUI application class.

    Uses Tkinter for the interface and OpenCV for video processing.
    """

    def __init__(self, root=None):
        self.root = root or tk.Tk()
        self.root.title("YOLO Wildlife Activity Monitor")

        # Video capture / source
        self.cap = None
        self.video_source_path = None

        # Video writer for recording
        self.video_writer = None
        self.record = tk.BooleanVar(value=False)

        # Processing state
        self.running = False
        self.detection_enabled = tk.BooleanVar(value=True)
        self.filter_var = tk.StringVar(value="None")
        self.min_area_var = tk.IntVar(value=0)  # used as extra area filter for YOLO boxes

        # Statistics and logging
        self.stats = StatsTracker()
        self.logger = DetectionLogger("wildlife_log.csv")

        # Frame geometry (initialized after opening source)
        self.frame_width = 640
        self.frame_height = 480

        # Core modules (initialized once we know frame size)
        self.detector = None
        self.filter_mgr = FrameFilter(self.filter_var.get())
        self.behavior = BehaviorAnalyzer()
        self.heatmap = None
        self.tracker = None

        # GUI elements to reference
        self.video_label = None
        self.status_var = tk.StringVar(value="Idle")
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.frame_var = tk.StringVar(value="Frames: 0")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0.0 s")
        self.detection_var = tk.StringVar(value="Detections: 0")

        # Internal bookkeeping
        self.frame_index = 0
        self.last_output_frame = None

        # Build interface and start loop
        self._build_ui()
        self._build_menu()
        self._bind_shortcuts()
        self.root.after(30, self._update_loop)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Create main layout."""
        main = ttk.Frame(self.root, padding=5)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)

        # Video display
        video_frame = ttk.Frame(main, borderwidth=2, relief="sunken")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Control panel
        control = ttk.Frame(main)
        control.grid(row=0, column=1, sticky="nsew")
        for r in range(10):
            control.rowconfigure(r, weight=0)
        control.rowconfigure(9, weight=1)
        control.columnconfigure(0, weight=1)
        control.columnconfigure(1, weight=1)

        # Start / Stop buttons
        ttk.Button(control, text="Start", command=self._on_start).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(control, text="Stop", command=self._on_stop).grid(
            row=0, column=1, sticky="ew"
        )

        # Open video
        ttk.Button(
            control,
            text="Open Video...",
            command=self._on_open_video,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)

        # Recording toggle
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.record,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        # Detection toggle
        ttk.Checkbutton(
            control,
            text="Enable Detection",
            variable=self.detection_enabled,
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        # Filter selection
        ttk.Label(control, text="Filter:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        filter_cb = ttk.Combobox(
            control,
            textvariable=self.filter_var,
            values=["None", "Grayscale", "Blur", "Edge"],
            state="readonly",
        )
        filter_cb.grid(row=4, column=1, sticky="ew", pady=(8, 0))
        filter_cb.bind("<<ComboboxSelected>>", lambda e: self._on_filter_change())

        # Min area slider (now used as YOLO area filter)
        ttk.Label(control, text="Min Box Area:").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        area_scale = ttk.Scale(
            control,
            from_=0,
            to=20000,
            variable=self.min_area_var,
            orient="horizontal",
            command=lambda v: self._on_min_area_change(float(v)),
        )
        area_scale.grid(row=5, column=1, sticky="ew", pady=(8, 0))

        # Statistics labels
        stats_frame = ttk.LabelFrame(control, text="Statistics")
        stats_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(stats_frame, textvariable=self.fps_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(stats_frame, textvariable=self.frame_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(stats_frame, textvariable=self.elapsed_var).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(stats_frame, textvariable=self.detection_var).grid(
            row=3, column=0, sticky="w"
        )

        # Snapshot / heatmap buttons
        ttk.Button(
            control,
            text="Save Snapshot",
            command=self._on_save_snapshot,
        ).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Button(
            control,
            text="Save Heatmap",
            command=self._on_save_heatmap,
        ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # Status bar
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        )
        status_bar.grid(row=1, column=0, sticky="ew")

    def _build_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video...", command=self._on_open_video)
        file_menu.add_command(label="Save Snapshot", command=self._on_save_snapshot)
        file_menu.add_command(
            label="Export Object Stats...", command=self._on_export_object_stats
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _bind_shortcuts(self):
        """Keyboard shortcuts."""
        self.root.bind("<Control-o>", lambda e: self._on_open_video())
        self.root.bind("<Control-s>", lambda e: self._on_save_snapshot())
        self.root.bind("<space>", self._on_space_toggle)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _on_start(self):
        self.running = True
        self.status_var.set("Running")

    def _on_stop(self):
        self.running = False
        self.status_var.set("Paused")

    def _on_space_toggle(self, _event):
        if self.running:
            self._on_stop()
        else:
            self._on_start()

    def _on_filter_change(self):
        self.filter_mgr.set_mode(self.filter_var.get())
        self.status_var.set(f"Filter: {self.filter_var.get()}")

    def _on_min_area_change(self, value):
        if self.detector is not None:
            self.detector.set_min_area(int(value))

    def _on_open_video(self):
        """Open a video file using a file dialog."""
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.video_source_path = path
        self._open_capture(path)
        self.status_var.set(f"Opened video: {os.path.basename(path)}")

    def _open_capture(self, source):
        """Open a cv2.VideoCapture from a path or camera index."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video source: {source}")
            self.cap = None
            return

        # Update geometry based on this source
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.frame_width)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.frame_height)
        if w > 0 and h > 0:
            self.frame_width = w
            self.frame_height = h

        # Recreate modules that depend on frame size
        self.detector = WildlifeDetector(
            min_area=self.min_area_var.get(),
            model_path="yolov8n.pt",
            conf_threshold=0.35,
        )
        self.heatmap = HeatmapGenerator(self.frame_width, self.frame_height)
        self.tracker = ObjectTracker(
            self.frame_width,
            self.frame_height,
            max_distance=150.0,  # allow bigger jumps between frames
            max_missed=20,  # tolerate short YOLO dropouts
            min_iou=0.2,  # require some overlap
        )

        # Reset stats and frame index
        self.stats = StatsTracker()
        self.frame_index = 0

    def _show_about(self):
        """Show About dialog."""
        messagebox.showinfo(
            "About",
            "YOLO Wildlife Activity Monitor\n\n"
            "Demonstration app for YOLO-based detection,\n"
            "object tracking and statistics.\n",
        )

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------
    def _update_loop(self):
        """Main GUI/video update loop."""
        if self.running:
            # If we have no capture, try to open default camera
            if self.cap is None:
                self._open_capture(0)

            frame = None
            ret = False
            if self.cap is not None:
                ret, frame = self.cap.read()

            if not ret or frame is None:
                self.running = False
                self.status_var.set("No more frames / capture closed")
            else:
                self.frame_index += 1
                self._process_frame(frame)

        self.root.after(20, self._update_loop)

    def _process_frame(self, frame):
        """Run detection, tracking, overlays and display for one frame."""
        # Ensure modules exist for unexpected sources
        if self.detector is None or self.heatmap is None or self.tracker is None:
            self.detector = WildlifeDetector(
                min_area=self.min_area_var.get(),
                model_path="yolov8n.pt",
                conf_threshold=0.35,
            )
            self.heatmap = HeatmapGenerator(self.frame_width, self.frame_height)
            self.tracker = ObjectTracker(
                self.frame_width,
                self.frame_height,
                max_distance=150.0,  # allow bigger jumps between frames
                max_missed=20,  # tolerate short YOLO dropouts
                min_iou=0.2,  # require some overlap
            )

        # Apply optional filter
        frame = self.filter_mgr.apply(frame)

        detection_count = 0

        if self.detection_enabled.get():
            detections = self.detector.detect(frame)  # list of {"box","label","conf"}

            if detections:
                boxes = [d["box"] for d in detections]
                tracks = self.tracker.update(boxes, self.frame_index)
                detection_count = len(tracks)

                # For each track, find nearest detection to reuse YOLO label/conf
                for track in tracks:
                    box = track["box"]
                    track_id = track["id"]
                    x, y, w, h = box

                    # Center of track box
                    tcx = x + w / 2.0
                    tcy = y + h / 2.0

                    best_det = None
                    best_dist = float("inf")
                    for det in detections:
                        dx, dy, dw, dh = det["box"]
                        dcx = dx + dw / 2.0
                        dcy = dy + dh / 2.0
                        d2 = (dcx - tcx) ** 2 + (dcy - tcy) ** 2
                        if d2 < best_dist:
                            best_dist = d2
                            best_det = det

                    if best_det is not None:
                        label = best_det["label"]
                        conf = best_det["conf"]
                    else:
                        label = "object"
                        conf = 0.0

                    # Behavior analysis (still uses box + frame size)
                    behavior_info = self.behavior.analyze(
                        track, self.frame_width, self.frame_height
                    )
                    behavior_text = behavior_info["status"]

                    # Heatmap accumulation
                    self.heatmap.add_point(box)

                    # Draw detection box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw ID + YOLO label
                    text_label = f"ID {track_id} - {label} ({conf:.2f})"
                    cv2.putText(
                        frame,
                        text_label,
                        (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    # Behavior text below the box
                    cv2.putText(
                        frame,
                        behavior_text,
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
            else:
                # No detections this frame: still age tracks
                self.tracker.update([], self.frame_index)
        else:
            # Detection disabled, no new info for tracker
            self.tracker.update([], self.frame_index)

        # Update statistics
        self.stats.update()
        self.fps_var.set(f"FPS: {self.stats.fps:.2f}")
        self.frame_var.set(f"Frames: {self.frame_index}")
        self.elapsed_var.set(f"Elapsed: {self.stats.elapsed:.1f} s")
        self.detection_var.set(f"Detections: {detection_count}")

        # Recording
        self._handle_recording(frame)

        # Keep last processed frame for snapshots
        self.last_output_frame = frame.copy()

        # Logging
        if self.logger is not None:
            self.logger.log(
                frame_index=self.frame_index,
                detections=detection_count,
                fps=self.stats.fps,
                elapsed=self.stats.elapsed,
            )

        # Display in GUI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

    # ------------------------------------------------------------------
    # Recording / Snapshot / Heatmap / Stats export
    # ------------------------------------------------------------------
    def _handle_recording(self, frame):
        """Initialize/close writer as needed and write current frame if recording."""
        if self.record.get():
            if self.video_writer is None:
                os.makedirs("recordings", exist_ok=True)
                name = datetime.now().strftime("recording_%Y%m%d_%H%M%S.avi")
                path = os.path.join("recordings", name)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(
                    path,
                    fourcc,
                    max(1.0, self.stats.fps or 20.0),
                    (self.frame_width, self.frame_height),
                )
                self.status_var.set(f"Recording started: {path}")
            if self.video_writer is not None:
                self.video_writer.write(frame)
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                self.status_var.set("Recording stopped")

    def _on_save_snapshot(self):
        if self.last_output_frame is None:
            messagebox.showinfo("Info", "No frame available to save.")
            return

        os.makedirs("snapshots", exist_ok=True)
        name = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.png")
        path = os.path.join("snapshots", name)
        cv2.imwrite(path, self.last_output_frame)
        self.status_var.set(f"Snapshot saved: {path}")

    def _on_export_object_stats(self):
        """
        Export per-object tracking statistics (dwell time, path length, speed)
        to a CSV file. Uses a rough average FPS estimated from the StatsTracker.
        """
        avg_fps = 0.0
        elapsed = self.stats.elapsed
        if elapsed > 0 and self.stats.total_frames > 0:
            avg_fps = self.stats.total_frames / elapsed

        os.makedirs("stats", exist_ok=True)
        name = datetime.now().strftime("object_stats_%Y%m%d_%H%M%S.csv")
        path = os.path.join("stats", name)

        fps_arg = avg_fps if avg_fps > 0 else None
        self.tracker.export_object_summaries_csv(path, fps=fps_arg)

        self.status_var.set(f"Object stats exported: {path}")
        messagebox.showinfo("Object Stats", f"Per-object statistics saved to:\n{path}")

    def _on_save_heatmap(self):
        if self.heatmap is None:
            messagebox.showinfo("Heatmap", "No heatmap data yet.")
            return
        path = self.heatmap.save_heatmap()
        messagebox.showinfo("Heatmap", f"Heatmap saved at:\n{path}")
        self.status_var.set(f"Heatmap saved: {path}")

    # ------------------------------------------------------------------
    # Cleanup / main
    # ------------------------------------------------------------------
    def _on_close(self):
        """Clean up resources and close the app."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.logger is not None:
            self.logger.close()
        self.root.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = VisionVideoApp()
    app.run()
