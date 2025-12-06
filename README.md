# Wildlife Object Detection & Tracking System

This project detects and tracks animals or objects in video using Python.  
It uses YOLOv8 for detection and a custom tracking system to follow each object with a stable ID.  
A small GUI lets anyone open a video and see detections, tracking paths, and basic behavior (moving / slow / stationary).

---

##  Features 

- **Object Detection** (YOLOv8)
- **Kalman Object Tracking** (keeps the same ID across frames)
- **Behavior Analysis** (fast / slow / still)
- **Heatmap** of where objects stay the most
- **GUI Application** (easy to use, no coding needed)

---

##  Why We Use a Kalman Filter

YOLO sometimes misses frames or gives shaky box positions.  
A **Kalman filter** helps by:

- Predicting where the object should be  
- Smoothing out noisy movement  
- Keeping the same ID even when YOLO briefly loses sight  

This makes tracking **much more stable and accurate**.

---

## 🛠 Install Requirements

Python 3.10+  
Install dependencies:

```bash
pip install ultralytics opencv-python pillow numpy matplotlib seaborn

"""
  How to Run the Program

open gui_app.py
This opens the full application with detection, tracking, behavior, toggle kalman filter and heatmap.


python gui_app.py
"""