# gui_app.py
import time
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from detector import MotionDetector
from drone import Drone
# from logger import DetectionLogger   # TODO: uncomment when logger.py is ready


class VisionDroneApp:
    """
    VisionDroneApp integrates:
    - Video streaming (camera or video file via OpenCV)
    - Motion detection
    - Drone simulation
    - Tkinter GUI with interactive controls
    - Optional CSV logging and video recording
    """

    def __init__(self, cam_index=0):
        # Current video source (camera index or file path string)
        self.source = cam_index
        self.source_is_file = False

        # Open initial camera
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open initial video source.")

        # Read one frame to obtain resolution
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame from source.")
        self.frame_height, self.frame_width = frame.shape[:2]

        # Initialize core modules
        self.detector = MotionDetector()
        self.drone = Drone(self.frame_width, self.frame_height)

        # TODO: create logger when logger.py is implemented
        # self.logger = DetectionLogger("detections.csv")

        # ------- Video recording state -------
        self.recording_enabled = tk.BooleanVar(value=False)
        self.video_writer = None

        # ------- Tkinter Setup -------
        self.root = tk.Tk()
        self.root.title("VisionDrone Tracking System")

        # GUI state variables
        self.mode_var = tk.StringVar(value="MANUAL")   # MANUAL or AUTO
        self.sensitivity_var = tk.IntVar(value=40)     # varThreshold
        self.min_area_var = tk.IntVar(value=1500)      # min contour area
        self.logging_enabled = tk.BooleanVar(value=False)  # CSV logging on/off

        self.running = True
        self.frame_id = 0
        self.last_time = time.time()
        self.fps = 0.0

        # Build UI components
        self._build_ui()

        # Keyboard control (W/A/S/D + Space)
        self.root.bind("<KeyPress>", self._on_key_press)

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start periodic update loop
        self._update_loop()

    # ---------------------------------------------
    # GUI CONSTRUCTION
    # ---------------------------------------------
    def _build_ui(self):
        """Build all GUI widgets and layout."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Video display label (left side)
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=7, padx=5, pady=5)

        # Control panel (right side)
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="n", padx=5, pady=5)

        # (1) Start / Stop buttons
        ttk.Button(control, text="Start", command=self._on_start).grid(row=0, column=0, sticky="ew")
        ttk.Button(control, text="Stop", command=self._on_stop).grid(row=0, column=1, sticky="ew")

        # (2) Mode dropdown (MANUAL / AUTO)
        ttk.Label(control, text="Mode:").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            control,
            textvariable=self.mode_var,
            values=["MANUAL", "AUTO"],
            state="readonly"
        ).grid(row=1, column=1, sticky="ew")

        # (3) Sensitivity slider (varThreshold)
        ttk.Label(control, text="Sensitivity").grid(row=2, column=0, sticky="w")
        ttk.Scale(
            control,
            from_=10,
            to=100,
            variable=self.sensitivity_var,
            command=self._on_sensitivity_change
        ).grid(row=2, column=1, sticky="ew")

        # (4) Min-area slider
        ttk.Label(control, text="Min Area").grid(row=3, column=0, sticky="w")
        ttk.Scale(
            control,
            from_=500,
            to=5000,
            variable=self.min_area_var,
            command=self._on_min_area_change
        ).grid(row=3, column=1, sticky="ew")

        # (5) Logging checkbox (only toggles flag for now)
        ttk.Checkbutton(
            control,
            text="Enable CSV Logging",
            variable=self.logging_enabled
        ).grid(row=4, column=0, columnspan=2, sticky="w")

        # (6) Reset drone button
        ttk.Button(
            control,
            text="Reset Drone",
            command=self._on_reset_drone
        ).grid(row=5, column=0, columnspan=2, sticky="ew")

        # (7) Open Video button (import video file)
        ttk.Button(
            control,
            text="Open Video...",
            command=self._on_open_video
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # (8) Record output checkbox (export processed video)
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.recording_enabled,
            command=self._on_record_toggle
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        # Status bar at bottom
        self.status_label = ttk.Label(main_frame, text="Ready", anchor="w")
        self.status_label.grid(row=8, column=0, columnspan=2, sticky="ew")

    # ---------------------------------------------
    # VIDEO SOURCE / RECORDING HELPERS
    # ---------------------------------------------
    def _open_source(self, source):
        """
        Open a new video source (camera index or file path).
        Recreates VideoCapture and resets frame counters.
        """
        # Release old capture
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from new source.")

        self.frame_height, self.frame_width = frame.shape[:2]

        # Recreate detector and drone with new resolution
        self.detector = MotionDetector()
        self.drone = Drone(self.frame_width, self.frame_height)

        self.frame_id = 0
        self.last_time = time.time()

    def _on_open_video(self):
        """
        Let user choose a video file and switch the input source to that file.
        """
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return  # user cancelled

        try:
            self._open_source(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video file:\n{e}")
            # fall back to previous source
            return

        self.source = path
        self.source_is_file = True
        self.status_label.config(text=f"Using video file: {path}")

    def _start_recording(self):
        """
        Ask for an output path and create a VideoWriter to record processed frames.
        """
        if self.video_writer is not None:
            return  # already recording

        save_path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")]
        )
        if not save_path:
            # User cancelled → uncheck the box
            self.recording_enabled.set(False)
            return

        # Try to get FPS from source; fall back to 30 if unknown
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps > 240:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            save_path,
            fourcc,
            fps,
            (self.frame_width, self.frame_height)
        )

        if not self.video_writer.isOpened():
            self.video_writer = None
            self.recording_enabled.set(False)
            messagebox.showerror("Error", "Failed to create output video file.")
            return

        self.status_label.config(text=f"Recording to: {save_path}")

    def _stop_recording(self):
        """Release VideoWriter and stop recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_label.config(text="Recording stopped.")

    def _on_record_toggle(self):
        """Handle Record Output Video checkbox state change."""
        if self.recording_enabled.get():
            self._start_recording()
        else:
            self._stop_recording()

    # ---------------------------------------------
    # CONTROL CALLBACKS
    # ---------------------------------------------
    def _on_start(self):
        """Start updating and processing frames."""
        self.running = True

    def _on_stop(self):
        """Pause the video processing loop."""
        self.running = False

    def _on_reset_drone(self):
        """Reset drone back to the center of the frame."""
        self.drone.reset_to_center()

    def _on_sensitivity_change(self, _):
        """Update motion detection sensitivity from GUI slider."""
        self.detector.set_var_threshold(self.sensitivity_var.get())

    def _on_min_area_change(self, _):
        """Update minimum contour area from GUI slider."""
        self.detector.set_min_area(self.min_area_var.get())

    def _on_key_press(self, event):
        """
        Manual drone control using keyboard:
        - W/A/S/D: move up/left/down/right
        - Space: enter SAFE mode
        """
        if self.mode_var.get() != "MANUAL":
            return

        step = self.drone.step
        key = event.keysym.lower()

        if key == "w":
            self.drone.move_manual(0, -step)
        elif key == "s":
            self.drone.move_manual(0, step)
        elif key == "a":
            self.drone.move_manual(-step, 0)
        elif key == "d":
            self.drone.move_manual(step, 0)
        elif key == "space":
            self.drone.mode = "SAFE"

    # ---------------------------------------------
    # MAIN UPDATE LOOP
    # ---------------------------------------------
    def _update_loop(self):
        """Tkinter periodic callback to grab and process video frames."""
        if self.running:
            ret, frame = self.cap.read()

            # For video files, when we hit the end, ret becomes False
            if not ret:
                if self.source_is_file:
                    # Stop when the file ends
                    self.running = False
                    self.status_label.config(text="End of video file.")
                else:
                    self.status_label.config(text="Camera read failed.")
                    self.running = False
            else:
                self._process_frame(frame)

        # Schedule the next update (in ~10 ms)
        self.root.after(10, self._update_loop)

    def _process_frame(self, frame):
        """Perform motion detection, update drone, and render to GUI."""
        self.frame_id += 1

        # Run motion detection
        boxes, mask = self.detector.detect(frame)

        # Use the largest box as the tracking target (if any)
        target_center = None
        if boxes:
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            target_center = (x + w // 2, y + h // 2)

        # Update drone mode based on GUI selection (unless already SAFE)
        if self.drone.mode != "SAFE":
            self.drone.mode = self.mode_var.get()

        # Auto-tracking behavior
        if self.drone.mode == "AUTO":
            self.drone.update_auto(target_center)

        # Safety check: switch to SAFE if near edges
        self.drone.check_safety(margin=30)

        # Draw detection boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw drone marker as a blue cross
        cv2.drawMarker(
            frame,
            (self.drone.x, self.drone.y),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            20,
            2
        )

        # Compute FPS
        now = time.time()
        dt = now - self.last_time
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self.last_time = now

        # TODO: enable CSV logging after logger is implemented
        # if self.logging_enabled.get():
        #     self.logger.log(self.frame_id, boxes, self.drone)

        # Write processed frame to video file if recording
        if self.recording_enabled.get() and self.video_writer is not None:
            self.video_writer.write(frame)

        # Convert frame (BGR -> RGB) for Tkinter display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update Tkinter label with the new frame
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Update status bar text
        self.status_label.config(
            text=f"Mode: {self.drone.mode} | Safe: {self.drone.safe} | "
                 f"Detections: {len(boxes)} | FPS: {self.fps:.1f}"
        )

    # ---------------------------------------------
    # CLEANUP
    # ---------------------------------------------
    def _on_close(self):
        """Handle window closing event: release resources and exit."""
        self.running = False

        if self.cap is not None:
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        # TODO: close logger once you actually create self.logger
        # if hasattr(self, "logger"):
        #     self.logger.close()

        self.root.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()
