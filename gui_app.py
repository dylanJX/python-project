# gui_app.py
"""
GUI application for wildlife-style drone monitoring.

Features:
- Display a live camera or selected video file
- Record (export) the processed video to a file
- Wildlife-style motion detection using background subtraction
- Simple image filters (None / Grayscale / Blur / Edge)
- Save snapshots of the current frame
- Display statistics (FPS, total frames, elapsed time, detections)
- Optional CSV logging of detections and stats
- Menu bar and keyboard shortcuts (Ctrl+O, Ctrl+S, Space)
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


class VisionVideoApp:
    """
    Main GUI application that orchestrates video I/O,
    wildlife detection, filtering, statistics and logging.
    """

    def __init__(self, cam_index: int = 0):
        """Initialize application state and GUI."""

        # -------------------------------------------------------
        # Create Tk root BEFORE any Tk variables
        # -------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("Wildlife Drone Monitoring Demo")

        # Tk variables
        self.recording_enabled = tk.BooleanVar(value=False)
        self.logging_enabled = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.detection_enabled = tk.BooleanVar(value=False)
        self.min_area_var = tk.IntVar(value=1500)
        self.filter_var = tk.StringVar(value="None")

        # -------------------------------------------------------
        # Video source initialization
        # -------------------------------------------------------
        self.source = cam_index
        self.source_is_file = False

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open initial camera source.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from camera.")

        self.frame_height, self.frame_width = frame.shape[:2]

        # VideoWriter for exporting processed video
        self.video_writer = None

        # Runtime flags
        self.running = True
        self.last_output_frame = None
        self.frame_index = 0

        # -------------------------------------------------------
        # Helper modules
        # -------------------------------------------------------
        self.detector = WildlifeDetector(min_area=self.min_area_var.get())
        self.filter_mgr = FrameFilter(mode=self.filter_var.get())
        self.stats = StatsTracker()
        self.logger = DetectionLogger("wildlife_log.csv")

        # Build GUI widgets and menu
        self._build_ui()
        self._build_menu()

        # Keyboard shortcuts
        self._bind_shortcuts()

        # Close-window protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start periodic update loop
        self.root.after(10, self._update_loop)

    # ---------------------------------------------------------
    # GUI construction
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        """Create and place all GUI widgets."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left: video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=20, padx=5, pady=5)

        # Right: control panel
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="n", padx=5, pady=5)

        # Row 0: Start / Stop
        ttk.Button(control, text="Start", command=self._on_start).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(control, text="Stop", command=self._on_stop).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )

        # Row 1: Open video file
        ttk.Button(control, text="Open Video...", command=self._on_open_video).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=4
        )

        # Row 2: Record checkbox
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.recording_enabled,
            command=self._on_record_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=2, pady=4)

        # Row 3: CSV logging checkbox
        ttk.Checkbutton(
            control,
            text="Enable CSV Logging",
            variable=self.logging_enabled,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=2, pady=2)

        # Row 4–6: Wildlife detection controls
        ttk.Label(control, text="Wildlife Detection", font=("Arial", 10, "bold")).grid(
            row=4, column=0, columnspan=2, pady=(8, 2)
        )

        ttk.Checkbutton(
            control,
            text="Enable Wildlife Detection",
            variable=self.detection_enabled,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=2, pady=2)

        ttk.Label(control, text="Min Area (pixels):").grid(row=6, column=0, sticky="w")
        self.min_area_scale = ttk.Scale(
            control,
            from_=500,
            to=5000,
            variable=self.min_area_var,
            orient="horizontal",
            command=self._on_min_area_change,
        )
        self.min_area_scale.grid(row=6, column=1, sticky="ew", padx=2, pady=2)

        # Row 7–8: Image filter selection
        ttk.Label(control, text="Image Filter:").grid(row=7, column=0, sticky="w")
        self.filter_combo = ttk.Combobox(
            control,
            textvariable=self.filter_var,
            values=["None", "Grayscale", "Blur", "Edge"],
            state="readonly",
        )
        self.filter_combo.grid(row=7, column=1, sticky="ew", padx=2, pady=2)
        self.filter_combo.current(0)
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        # Row 9: Save snapshot button
        ttk.Button(
            control,
            text="Save Snapshot",
            command=self._on_save_snapshot,
        ).grid(row=8, column=0, columnspan=2, sticky="ew", padx=2, pady=4)

        # Row 10–13: Statistics (FPS, detections, frames, elapsed time)
        ttk.Label(control, text="FPS:").grid(row=9, column=0, sticky="w")
        self.fps_label = ttk.Label(control, text="0.0")
        self.fps_label.grid(row=9, column=1, sticky="e")

        ttk.Label(control, text="Detections:").grid(row=10, column=0, sticky="w")
        self.detection_label = ttk.Label(control, text="0")
        self.detection_label.grid(row=10, column=1, sticky="e")

        ttk.Label(control, text="Frames:").grid(row=11, column=0, sticky="w")
        self.frames_label = ttk.Label(control, text="0")
        self.frames_label.grid(row=11, column=1, sticky="e")

        ttk.Label(control, text="Elapsed (s):").grid(row=12, column=0, sticky="w")
        self.elapsed_label = ttk.Label(control, text="0.0")
        self.elapsed_label.grid(row=12, column=1, sticky="e")

        # Status bar at bottom
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
        )
        status_bar.grid(row=20, column=0, columnspan=2, sticky="ew")

    def _build_menu(self) -> None:
        """Create top menu bar (File / Help)."""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Open Video...",
            command=self._on_open_video,
            accelerator="Ctrl+O",
        )
        file_menu.add_command(
            label="Save Snapshot",
            command=self._on_save_snapshot,
            accelerator="Ctrl+S",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
            command=self._on_close,
        )
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts for common actions."""
        # Ctrl+O to open video file
        self.root.bind("<Control-o>", lambda event: self._on_open_video())
        # Ctrl+S to save snapshot
        self.root.bind("<Control-s>", lambda event: self._on_save_snapshot())
        # Space to toggle start/stop
        self.root.bind("<space>", self._on_space_toggle)

    # ---------------------------------------------------------
    # Control callbacks
    # ---------------------------------------------------------
    def _on_start(self) -> None:
        """Resume the video update loop."""
        self.running = True
        self.status_var.set("Running")

    def _on_stop(self) -> None:
        """Pause the video update loop."""
        self.running = False
        self.status_var.set("Paused")

    def _on_space_toggle(self, _event) -> None:
        """Toggle between running and paused using the space bar."""
        if self.running:
            self._on_stop()
        else:
            self._on_start()

    def _on_open_video(self) -> None:
        """Ask user for a video file and open it as the new source."""
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video:\n{path}")
            # Try to fall back to original camera
            self.cap = cv2.VideoCapture(self.source)
            return

        # Update frame size from the new source
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self.frame_width, self.frame_height = w, h

        # Reset background model when changing source
        self.detector.reset_model()

        self.source = path
        self.source_is_file = True
        self.status_var.set(f"Using video file: {path}")

    def _on_record_toggle(self) -> None:
        """Start or stop recording based on the checkbox state."""
        if self.recording_enabled.get():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        """Create a VideoWriter for exporting the processed video."""
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
            self.recording_enabled.set(False)
            self.video_writer = None
            messagebox.showerror("Error", "Failed to create output video file.")
            return

        self.status_var.set(f"Recording to: {save_path}")

    def _stop_recording(self) -> None:
        """Stop recording and release the VideoWriter."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_var.set("Recording stopped")

    def _on_min_area_change(self, _event) -> None:
        """Update detector's minimum area from the scale widget."""
        self.detector.set_min_area(self.min_area_var.get())

    def _on_filter_change(self, _event) -> None:
        """Update the current filter mode from the combobox."""
        self.filter_mgr.set_mode(self.filter_var.get())

    def _on_save_snapshot(self) -> None:
        """Save the last processed frame to the 'snapshots' folder."""
        if self.last_output_frame is None:
            messagebox.showinfo("Info", "No frame available to save yet.")
            return

        os.makedirs("snapshots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("snapshots", f"snapshot_{timestamp}.png")
        cv2.imwrite(filename, self.last_output_frame)
        self.status_var.set(f"Snapshot saved: {filename}")

    def _show_about(self) -> None:
        """Show a simple About dialog with project information."""
        message = (
            "VisionDrone: Intelligent Tracking System\n\n"
            "Wildlife Drone Monitoring Demo\n"
            "Developed by: Zhelin Zheng & Jingxuan Zhu\n\n"
            "Features:\n"
            "- Video import/export\n"
            "- Wildlife-style motion detection\n"
            "- Image filters and statistics\n"
            "- CSV logging and snapshots"
        )
        messagebox.showinfo("About", message)

    # ---------------------------------------------------------
    # Main loop and processing
    # ---------------------------------------------------------
    def _update_loop(self) -> None:
        """Main periodic update loop scheduled with root.after()."""
        if self.running:
            ret, frame = self.cap.read()

            if not ret:
                if self.source_is_file:
                    self.running = False
                    self.status_var.set("End of video file")
                else:
                    self.running = False
                    self.status_var.set("Camera read failed")
            else:
                self._process_frame(frame)

        self.root.after(10, self._update_loop)

    def _process_frame(self, frame) -> None:
        """
        Process a single frame:
        - Update statistics
        - Optionally run wildlife detection
        - Apply selected image filter
        - Optionally record to video file
        - Display in Tkinter GUI
        - Optionally log stats and detections to CSV
        """
        # ---------------- Statistics ----------------
        self.frame_index += 1
        self.stats.update()

        # Update statistics labels
        self.fps_label.config(text=f"{self.stats.fps:.1f}")
        self.frames_label.config(text=str(self.stats.total_frames))
        self.elapsed_label.config(text=f"{self.stats.elapsed:.1f}")

        # ---------------- Wildlife detection ----------------
        detection_count = 0
        if self.detection_enabled.get():
            # Get all detected boxes from the detector
            boxes = self.detector.detect(frame)

            # Only draw the largest box to avoid many small rectangles
            if boxes:
                largest_box = max(boxes, key=lambda b: b[2] * b[3])
                boxes_to_draw = [largest_box]
            else:
                boxes_to_draw = []

            detection_count = len(boxes_to_draw)

            for i, (x, y, w, h) in enumerate(boxes_to_draw, start=1):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Animal {i}",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        self.detection_label.config(text=str(detection_count))

        # ---------------- Image filter ----------------
        frame = self.filter_mgr.apply(frame)

        # Store last output frame for snapshot
        self.last_output_frame = frame.copy()

        # ---------------- Recording ----------------
        if self.recording_enabled.get() and self.video_writer is not None:
            self.video_writer.write(frame)

        # ---------------- CSV logging ----------------
        if self.logging_enabled.get():
            self.logger.log(
                frame_index=self.frame_index,
                detections=detection_count,
                fps=self.stats.fps,
                elapsed=self.stats.elapsed,
            )

        # ---------------- Display in Tkinter ----------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def _on_close(self) -> None:
        """Release resources and close the window."""
        self.running = False

        if self.cap is not None:
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        if self.logger is not None:
            self.logger.close()

        self.root.destroy()

    def run(self) -> None:
        """Start Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = VisionVideoApp()
    app.run()
