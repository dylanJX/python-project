# gui_app.py
import time
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# from detector import MotionDetector
# from drone import Drone

#from wildlife_data import WildlifeDataset
# from logger import DetectionLogger   # Also disabled for now


class VisionDroneApp:
    """
    A simplified GUI application that supports:
    - Camera or video file streaming
    - Optional video recording
    - Wildlife dataset loading and summary display
    The MotionDetector and Drone modules are currently disabled.
    """

    def __init__(self, cam_index=0):
        # Store initial video source
        self.source = cam_index
        self.source_is_file = False

        # Open the initial camera
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open the initial camera.")

        # Read a test frame to determine resolution
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame from the camera.")
        self.frame_height, self.frame_width = frame.shape[:2]

        # ---------------------------------------------------------
        # MotionDetector / Drone temporarily disabled
        # ---------------------------------------------------------
        # self.detector = MotionDetector()
        # self.drone = Drone(self.frame_width, self.frame_height)

        # Wildlife dataset (None until user loads a CSV file)
        self.dataset: WildlifeDataset | None = None

        # Recording settings
        self.recording_enabled = tk.BooleanVar(value=False)
        self.video_writer = None

        # Tkinter root window
        self.root = tk.Tk()
        self.root.title("VisionDrone Tracking System")

        # GUI state variables (some unused until detector/drone are re-enabled)
        self.mode_var = tk.StringVar(value="MANUAL")
        self.sensitivity_var = tk.IntVar(value=40)
        self.min_area_var = tk.IntVar(value=1500)
        self.logging_enabled = tk.BooleanVar(value=False)

        # Wildlife dataset GUI variable
        self.species_var = tk.StringVar(value="")

        # Runtime state
        self.running = True
        self.frame_id = 0
        self.last_time = time.time()
        self.fps = 0.0

        # Build all GUI elements
        self._build_ui()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start the update loop
        self.root.after(10, self._update_loop)

    # -------------------------------------------------------------
    # GUI Construction
    # -------------------------------------------------------------
    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left side video display panel
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, rowspan=20, padx=5, pady=5)

        # Right side control panel
        control = ttk.Frame(main_frame)
        control.grid(row=0, column=1, sticky="n", padx=5, pady=5)

        # Start / Stop buttons
        ttk.Button(control, text="Start", command=self._on_start).grid(row=0, column=0, sticky="ew")
        ttk.Button(control, text="Stop", command=self._on_stop).grid(row=0, column=1, sticky="ew")

        # Open video file button
        ttk.Button(control, text="Open Video...", command=self._on_open_video).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=5
        )

        # Video recording checkbox
        ttk.Checkbutton(
            control,
            text="Record Output Video",
            variable=self.recording_enabled,
            command=self._on_record_toggle
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        # Wildlife dataset section
        ttk.Label(control, text="Wildlife Dataset", font=("Arial", 11, "bold")).grid(
            row=3, column=0, columnspan=2, pady=(10, 5)
        )

        ttk.Button(
            control,
            text="Load Wildlife CSV",
            command=self._on_load_data
        ).grid(row=4, column=0, columnspan=2, sticky="ew")

        # Species selection
        ttk.Label(control, text="Species:").grid(row=5, column=0, sticky="w")
        self.species_combo = ttk.Combobox(
            control,
            textvariable=self.species_var,
            values=[],
            state="readonly"
        )
        self.species_combo.grid(row=5, column=1, sticky="ew")

        ttk.Button(
            control,
            text="Show Species Stats",
            command=self._on_show_stats
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", anchor="w")
        self.status_label.grid(row=21, column=0, columnspan=2, sticky="ew")

    # -------------------------------------------------------------
    # Wildlife Dataset Functions
    # -------------------------------------------------------------
    def _on_load_data(self):
        """Allow user to select a wildlife CSV file and load it."""
        path = filedialog.askopenfilename(
            title="Select wildlife CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            # Load up to 3000 rows to keep performance reasonable
            self.dataset = WildlifeDataset(path, max_rows=3000)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
            self.dataset = None
            return

        # Populate species dropdown
        species_list = self.dataset.get_species_list()
        self.species_combo["values"] = species_list

        if species_list:
            self.species_var.set(species_list[0])

        self.status_label.config(text=f"Dataset loaded: {path}")

    def _on_show_stats(self):
        """Display summary statistics for the selected species."""
        if self.dataset is None:
            messagebox.showinfo("Info", "Please load a wildlife dataset first.")
            return

        species = self.species_var.get().strip()
        if not species:
            messagebox.showinfo("Info", "Please select a species.")
            return

        num_obs, total_count = self.dataset.species_summary(species)

        messagebox.showinfo(
            "Species Statistics",
            f"Species: {species}\n"
            f"Observations: {num_obs}\n"
            f"Total Count: {total_count}"
        )

    # -------------------------------------------------------------
    # Video Source Handling
    # -------------------------------------------------------------
    def _on_open_video(self):
        """Allow the user to select a video file as input."""
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        # Release previous capture
        if self.cap is not None:
            self.cap.release()

        # Open the selected file
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        self.source_is_file = True
        self.status_label.config(text=f"Using video file: {path}")

    # -------------------------------------------------------------
    # Video Recording
    # -------------------------------------------------------------
    def _on_record_toggle(self):
        """Start or stop video recording."""
        if self.recording_enabled.get():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Open a file dialog to select output video filename."""
        save_path = filedialog.asksaveasfilename(
            title="Save output video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )
        if not save_path:
            self.recording_enabled.set(False)
            return

        # Determine FPS
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
            messagebox.showerror("Error", "Failed to create output video file.")
        else:
            self.status_label.config(text=f"Recording to: {save_path}")

    def _stop_recording(self):
        """Stop recording and release writer."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.status_label.config(text="Recording stopped.")

    # -------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------
    def _update_loop(self):
        """Main GUI loop — reads frames and updates the display."""
        if self.running:
            ret, frame = self.cap.read()

            # If reading from file and end is reached
            if not ret:
                if self.source_is_file:
                    self.running = False
                    self.status_label.config(text="Reached end of video file.")
                else:
                    self.status_label.config(text="Camera read failed.")
                    self.running = False
            else:
                self._process_frame(frame)

        # Schedule next GUI update
        self.root.after(10, self._update_loop)

    def _process_frame(self, frame):
        """Process each frame (detection & drone logic disabled)."""

        # -----------------------------------------------------
        # MotionDetector / Drone logic temporarily removed:
        # -----------------------------------------------------
        # boxes, mask = self.detector.detect(frame)
        # if boxes:
        #     x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        #     target_center = (x + w // 2, y + h // 2)
        # else:
        #     target_center = None
        #
        # if self.drone.mode != "SAFE":
        #     self.drone.mode = self.mode_var.get()
        #
        # if self.drone.mode == "AUTO":
        #     self.drone.update_auto(target_center)
        #
        # self.drone.check_safety(margin=30)
        #
        # for (x, y, w, h) in boxes:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # cv2.drawMarker(frame, (self.drone.x, self.drone.y), (255, 0, 0),
        #                cv2.MARKER_CROSS, 20, 2)

        # Write frame if recording
        if self.recording_enabled.get() and self.video_writer is not None:
            self.video_writer.write(frame)

        # Convert to Tkinter image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # -------------------------------------------------------------
    # Window Close
    # -------------------------------------------------------------
    def _on_close(self):
        """Release all resources and close the window."""
        self.running = False

        if self.cap is not None:
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        self.root.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()
