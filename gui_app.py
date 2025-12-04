# gui_app.py
"""
This app:
- opens a camera or video file,
- runs simple motion-based wildlife detection,
- allows basic image filters,
- shows FPS and detection statistics,
- lets the user record video, save snapshots and export a heatmap.
"""


import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from PIL import Image, ImageTk

# project modules
from detector import WildlifeDetector
from image_filters import FrameFilter
from stats import StatsTracker
from logger import DetectionLogger
from classifier import classify
from behavior import BehaviorAnalyzer
from heatmap import HeatmapGenerator


class VisionVideoApp:
    """
    Main GUI application that integrates:
    - Video input/output
    - Detection + classification + behavior analysis
    - Image filtering
    - Statistics and CSV logging
    """

    def __init__(self, cam_index: int = 0):
        """Initialize application state and GUI."""

        # Tk root must be created before any tk variables
        self.root = tk.Tk()
        self.root.title("Wildlife Drone Monitoring Demo")

        # Tk variables
        self.recording_enabled = tk.BooleanVar(value=False)
        self.logging_enabled = tk.BooleanVar(value=False)
        self.detection_enabled = tk.BooleanVar(value=False)
        self.min_area_var = tk.IntVar(value=1500)
        self.filter_var = tk.StringVar(value="None")
        self.status_var = tk.StringVar(value="Ready")

        # Video source
        self.source = cam_index
        self.source_is_file = False
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera or video source.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read initial frame.")

        self.frame_height, self.frame_width = frame.shape[:2]

        # Video writer (for exporting)
        self.video_writer = None
        self.running = True
        self.frame_index = 0
        self.last_output_frame = None

        # Core modules
        self.detector = WildlifeDetector(self.min_area_var.get())
        self.filter_mgr = FrameFilter(self.filter_var.get())
        self.stats = StatsTracker()
        self.logger = DetectionLogger("wildlife_log.csv")
        self.behavior = BehaviorAnalyzer()
        self.heatmap = HeatmapGenerator(self.frame_width, self.frame_height)

        # Build GUI components
        self._build_ui()
        self._build_menu()
        self._bind_shortcuts()

        # Exit handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start loop
        self.root.after(10, self._update_loop)

    # ---------------------------------------------------------
    # GUI layout
    # ---------------------------------------------------------
    def _build_ui(self):
        """Build all main widgets (video + controls)."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Video display area
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=22, padx=5, pady=5)

        # Control panel
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="n", padx=5, pady=5)

        # Start/Stop
        ttk.Button(control, text="Start", command=self._on_start).grid(
            row=0, column=0, sticky="ew")
        ttk.Button(control, text="Stop", command=self._on_stop).grid(
            row=0, column=1, sticky="ew")

        # Open video
        ttk.Button(control, text="Open Video...", command=self._on_open_video).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=4)

        # Recording toggle
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.recording_enabled,
            command=self._on_record_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        # CSV logging toggle
        ttk.Checkbutton(
            control,
            text="Enable CSV Logging",
            variable=self.logging_enabled,
        ).grid(row=3, column=0, columnspan=2, sticky="w")

        # Wildlife detection
        ttk.Label(control, text="Wildlife Detection", font=("Arial", 10, "bold")).grid(
            row=4, column=0, columnspan=2, pady=(8, 2)
        )

        ttk.Checkbutton(
            control,
            text="Enable Detection",
            variable=self.detection_enabled,
        ).grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(control, text="Min Area (pixels):").grid(row=6, column=0)
        ttk.Scale(
            control,
            from_=500,
            to=5000,
            variable=self.min_area_var,
            orient="horizontal",
            command=self._on_min_area_change,
        ).grid(row=6, column=1, sticky="ew")

        # Image filter
        ttk.Label(control, text="Image Filter:").grid(row=7, column=0)
        self.filter_combo = ttk.Combobox(
            control,
            textvariable=self.filter_var,
            values=["None", "Grayscale", "Blur", "Edge"],
            state="readonly",
        )
        self.filter_combo.grid(row=7, column=1, sticky="ew")
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        # Snapshot
        ttk.Button(
            control, text="Save Snapshot", command=self._on_save_snapshot
        ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)

        # Heatmap
        ttk.Button(
            control, text="Save Heatmap", command=self._on_save_heatmap
        ).grid(row=9, column=0, columnspan=2, sticky="ew", pady=4)

        # Statistics display
        ttk.Label(control, text="FPS:").grid(row=10, column=0)
        self.fps_label = ttk.Label(control, text="0.0")
        self.fps_label.grid(row=10, column=1, sticky="e")

        ttk.Label(control, text="Detections:").grid(row=11, column=0)
        self.detection_label = ttk.Label(control, text="0")
        self.detection_label.grid(row=11, column=1, sticky="e")

        ttk.Label(control, text="Frames:").grid(row=12, column=0)
        self.frames_label = ttk.Label(control, text="0")
        self.frames_label.grid(row=12, column=1, sticky="e")

        ttk.Label(control, text="Elapsed (s):").grid(row=13, column=0)
        self.elapsed_label = ttk.Label(control, text="0.0")
        self.elapsed_label.grid(row=13, column=1, sticky="e")

        ttk.Label(control, text="Behavior:").grid(row=14, column=0)
        self.behavior_label = ttk.Label(control, text="N/A")
        self.behavior_label.grid(row=14, column=1, sticky="e")

        # Status bar
        ttk.Label(
            main_frame, textvariable=self.status_var, anchor="w", relief="sunken"
        ).grid(row=22, column=0, columnspan=2, sticky="ew")

    def _build_menu(self):
        """Top menu bar."""
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video...", command=self._on_open_video)
        file_menu.add_command(label="Save Snapshot", command=self._on_save_snapshot)
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

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------
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

    def _on_open_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.cap.release()
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open: {path}")
            self.cap = cv2.VideoCapture(self.source)
            return

        # Update resolution
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self.frame_width, self.frame_height = w, h

        # Reset modules
        self.detector.reset_model()
        self.behavior = BehaviorAnalyzer()
        self.heatmap = HeatmapGenerator(self.frame_width, self.frame_height)

        self.source = path
        self.source_is_file = True
        self.status_var.set(f"Using video file: {path}")

    # ---------------------------------------------------------
    # Recording
    # ---------------------------------------------------------
    def _on_record_toggle(self):
        if self.recording_enabled.get():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        save_path = filedialog.asksaveasfilename(
            title="Save output video",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*"),
            ],
        )
        if not save_path:
            self.recording_enabled.set(False)
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            save_path, fourcc, fps, (self.frame_width, self.frame_height)
        )

        if not self.video_writer.isOpened():
            self.video_writer = None
            self.recording_enabled.set(False)
            messagebox.showerror("Error", "Failed to create output file.")
            return

        self.status_var.set(f"Recording to: {save_path}")

    def _stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_var.set("Recording stopped")

    def _on_min_area_change(self, _event=None):
        """
        Callback for the min-area slider.

        Reads the current value from self.min_area_var and
        updates the detector configuration.
        """
        self.detector.set_min_area(self.min_area_var.get())

    def _on_filter_change(self, _event=None):
        """
        Callback for the image filter combobox.

        Updates the current filter mode in FrameFilter.
        """
        self.filter_mgr.set_mode(self.filter_var.get())


    # ---------------------------------------------------------
    # Processing loop
    # ---------------------------------------------------------
    def _update_loop(self):
        if self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_is_file:
                    self.status_var.set("End of video file")
                else:
                    self.status_var.set("Camera read failed")
                self.running = False
            else:
                self._process_frame(frame)

        self.root.after(10, self._update_loop)

    def _process_frame(self, frame):
        """Process one frame: detection, filters, behavior, logging, display."""
        self.frame_index += 1
        self.stats.update()

        # Update GUI statistics
        self.fps_label.config(text=f"{self.stats.fps:.1f}")
        self.frames_label.config(text=str(self.stats.total_frames))
        self.elapsed_label.config(text=f"{self.stats.elapsed:.1f}")

        # -----------------------------------------------------
        # Wildlife detection / classification / behavior
        # -----------------------------------------------------
        detection_count = 0
        behavior_text = "N/A"

        largest_box = None
        if self.detection_enabled.get():
            boxes = self.detector.detect(frame)

            if boxes:
                largest_box = max(boxes, key=lambda b: b[2] * b[3])
                detection_count = 1

                # Classification
                species, conf = classify(largest_box)

                # Behavior analysis
                behavior_info = self.behavior.analyze(
                    largest_box, self.frame_width, self.frame_height
                )
                behavior_text = behavior_info["status"]

                # Heatmap accumulation
                self.heatmap.add_point(largest_box)

                # Draw detection box
                x, y, w, h = largest_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Text labels
                cv2.putText(frame, f"{species} ({conf})",
                            (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

                cv2.putText(frame, behavior_text,
                            (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)
            else:
                behavior_text = "No detection"

        self.detection_label.config(text=str(detection_count))
        self.behavior_label.config(text=behavior_text)

        # -----------------------------------------------------
        # Image filter
        # -----------------------------------------------------
        frame = self.filter_mgr.apply(frame)
        self.last_output_frame = frame.copy()

        # Recording
        if self.recording_enabled.get() and self.video_writer is not None:
            self.video_writer.write(frame)

        # CSV logging
        if self.logging_enabled.get():
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

    # ---------------------------------------------------------
    # Snapshot / Heatmap
    # ---------------------------------------------------------
    def _on_save_snapshot(self):
        if self.last_output_frame is None:
            messagebox.showinfo("Info", "No frame available to save.")
            return

        os.makedirs("snapshots", exist_ok=True)
        name = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.png")
        path = os.path.join("snapshots", name)
        cv2.imwrite(path, self.last_output_frame)
        self.status_var.set(f"Snapshot saved: {path}")

    def _on_save_heatmap(self):
        path = self.heatmap.save_heatmap()
        messagebox.showinfo("Heatmap", f"Heatmap saved at:\n{path}")
        self.status_var.set(f"Heatmap saved: {path}")

    # ---------------------------------------------------------
    # Closing
    # ---------------------------------------------------------
    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Wildlife Drone Monitoring System\n"
            "Developed by: Zhelin Zheng & Jingxuan Zhu\n"
            "Includes detection, classification, filters, heatmap,\n"
            "recording, and statistics."
        )

    def _on_close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.logger is not None:
            self.logger.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = VisionVideoApp()
    app.run()
