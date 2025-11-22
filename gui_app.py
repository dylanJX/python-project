# gui_app.py
"""
Simple GUI app for:
- Displaying a camera or selected video file
- Recording (exporting) the displayed video to a file
"""

import time
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class VisionVideoApp:
    """
    VisionVideoApp provides:
    - Camera / video file playback
    - Video recording/export
    - Basic Tkinter GUI
    """

    def __init__(self, cam_index: int = 0):
        """Initialize application state and GUI."""

        # -------------------------------------------------------
        # Create Tk root FIRST before creating any Tk variables
        # -------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("VisionDrone - Video Import/Export Demo")

        # Tkinter variables must be created AFTER Tk() exists
        self.recording_enabled = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

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

        # VideoWriter for exporting
        self.video_writer = None

        # Runtime state
        self.running = True
        self.last_time = time.time()
        self.fps = 0.0

        # Build GUI
        self._build_ui()

        # Close-window protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start update loop
        self.root.after(10, self._update_loop)

    # ---------------------------------------------------------
    # GUI construction
    # ---------------------------------------------------------
    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=5, pady=5)

        # Right panel
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="n", padx=5, pady=5)

        # Start/Stop
        ttk.Button(control, text="Start", command=self._on_start).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(control, text="Stop", command=self._on_stop).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )

        # Open video
        ttk.Button(control, text="Open Video...", command=self._on_open_video).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=5
        )

        # Record checkbox
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.recording_enabled,
            command=self._on_record_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=2, pady=5)

        # FPS display
        ttk.Label(control, text="FPS:").grid(row=3, column=0, sticky="w")
        self.fps_label = ttk.Label(control, text="0.0")
        self.fps_label.grid(row=3, column=1, sticky="e")

        # Status bar
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
        )
        status_bar.grid(row=10, column=0, columnspan=2, sticky="ew")

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------
    def _on_start(self):
        self.running = True
        self.status_var.set("Running")

    def _on_stop(self):
        self.running = False
        self.status_var.set("Paused")

    def _on_open_video(self):
        """Ask user for a video file and open it."""
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
            return

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
        """Create VideoWriter for exporting the processed video."""
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
            messagebox.showerror("Error", "Failed to create output video file.")
            return

        self.status_var.set(f"Recording to: {save_path}")

    def _stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_var.set("Recording stopped")

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    def _update_loop(self):
        """Main periodic update loop."""
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

    def _process_frame(self, frame):
        """Show frame and optionally record it."""
        # FPS calculation
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        self.fps_label.config(text=f"{self.fps:.1f}")

        # Record if enabled
        if self.recording_enabled.get() and self.video_writer is not None:
            self.video_writer.write(frame)

        # Convert to Tkinter image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update GUI label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def _on_close(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = VisionVideoApp()
    app.run()
