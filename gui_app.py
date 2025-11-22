# gui_app.py

import time
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class VisionVideoApp:
    """
    VisionVideoApp provides a basic Tkinter GUI that allows the user to:
    - Start/stop video playback
    - Switch between camera and a selected video file
    - Record (export) the currently displayed video to a file
    """

    def __init__(self, cam_index: int = 0):
        # Current video source: camera index or file path
        self.source = cam_index
        self.source_is_file = False

        # Open initial camera source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open initial camera source.")

        # Read first frame to get resolution
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame from camera.")
        self.frame_height, self.frame_width = frame.shape[:2]

        # Recording state
        self.recording_enabled = tk.BooleanVar(value=False)
        self.video_writer = None

        # Runtime flags and FPS tracking
        self.running = True
        self.last_time = time.time()
        self.fps = 0.0

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("VisionDrone - Video Import/Export Demo")

        # Status text for bottom bar
        self.status_var = tk.StringVar(value="Ready")

        # Build all GUI widgets
        self._build_ui()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start the periodic update loop
        self.root.after(10, self._update_loop)

    # ---------------------------------------------------------
    # GUI construction
    # ---------------------------------------------------------
    def _build_ui(self) -> None:
        """Create and place all GUI widgets."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left side: video display
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=5, pady=5)

        # Right side: control panel
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="n", padx=5, pady=5)

        # Row 0: start / stop
        ttk.Button(control, text="Start", command=self._on_start).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(control, text="Stop", command=self._on_stop).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )

        # Row 1: open video file (import)
        ttk.Button(control, text="Open Video...", command=self._on_open_video).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=4
        )

        # Row 2: recording checkbox (export)
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.recording_enabled,
            command=self._on_record_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=2, pady=4)

        # Row 3: simple FPS label
        ttk.Label(control, text="FPS:").grid(row=3, column=0, sticky="w")
        self.fps_label = ttk.Label(control, text="0.0")
        self.fps_label.grid(row=3, column=1, sticky="e")

        # Bottom: status bar across the whole window
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
        )
        status_bar.grid(row=10, column=0, columnspan=2, sticky="ew")

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

    def _on_open_video(self) -> None:
        """
        Let the user select a video file and use it as the new source.
        The previous camera/file capture is released.
        """
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return  # User canceled

        # Release previous capture if any
        if self.cap is not None:
            self.cap.release()

        # Open the selected file
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video file:\n{path}")
            # Try to reopen camera as fallback
            self.cap = cv2.VideoCapture(self.source)
            return

        # Update internal flags
        self.source = path
        self.source_is_file = True
        self.status_var.set(f"Using video file: {path}")

    # ---------------------------------------------------------
    # Recording (export) handling
    # ---------------------------------------------------------
    def _on_record_toggle(self) -> None:
        """Start or stop recording depending on the checkbox."""
        if self.recording_enabled.get():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        """
        Ask the user for an output filename and create a VideoWriter.
        Frames will be written in _process_frame() when recording is enabled.
        """
        if self.video_writer is not None:
            # Already recording
            return

        save_path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*"),
            ],
        )
        if not save_path:
            # User canceled; uncheck recording
            self.recording_enabled.set(False)
            return

        # Try to obtain FPS from current source
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps > 240:
            # Fallback to a reasonable default
            fps = 30.0

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            save_path,
            fourcc,
            fps,
            (self.frame_width, self.frame_height),
        )

        if not self.video_writer.isOpened():
            self.video_writer = None
            self.recording_enabled.set(False)
            messagebox.showerror("Error", "Failed to create output video file.")
            return

        self.status_var.set(f"Recording to: {save_path}")

    def _stop_recording(self) -> None:
        """Release the VideoWriter and stop recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_var.set("Recording stopped")

    # ---------------------------------------------------------
    # Main update loop
    # ---------------------------------------------------------
    def _update_loop(self) -> None:
        """
        Periodic callback scheduled with root.after().
        Reads frames when running and updates the display.
        """
        if self.running:
            ret, frame = self.cap.read()

            if not ret:
                # If reading from a file and we reach the end, stop playback
                if self.source_is_file:
                    self.running = False
                    self.status_var.set("End of video file")
                else:
                    self.running = False
                    self.status_var.set("Camera read failed")
            else:
                self._process_frame(frame)

        # Schedule the next update in ~10 ms
        self.root.after(10, self._update_loop)

    def _process_frame(self, frame) -> None:
        """
        Handle a single video frame:
        - Update FPS estimation
        - Write to output file if recording is enabled
        - Convert and show the frame in the Tkinter GUI
        """
        # FPS computation
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        self.fps_label.config(text=f"{self.fps:.1f}")

        # Write frame to video file if recording is active
        if self.recording_enabled.get() and self.video_writer is not None:
            self.video_writer.write(frame)

        # Convert frame (BGR -> RGB) and display in Tkinter label
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Keep a reference to avoid garbage collection
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def _on_close(self) -> None:
        """Cleanly release resources and close the window."""
        self.running = False

        if self.cap is not None:
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        self.root.destroy()

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = VisionVideoApp(cam_index=0)
    app.run()
