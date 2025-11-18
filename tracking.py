import cv2
import numpy as np
import time

def main(
    cam_index=0,
    min_area=1500,            # ignore tiny motions (pixels)
    history=500,              # how long the background model "remembers"
    var_threshold=40,         # sensitivity of foreground detection (lower = more sensitive)
    show_fps=True
):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: could not open camera index {cam_index}")
        return

    # Optional: set resolution (uncomment/change if needed)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create background subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=True
    )

    # Morphological kernel to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    prev_time = time.time()
    fps = 0.0
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: frame grab failed.")
            break

        # Optional slight blur helps reduce high-frequency noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Foreground mask where motion is detected
        fgmask = subtractor.apply(blurred)

        # Remove shadows (in MOG2, shadows are gray ≈ 127). Keep only strong foreground (255)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Morphology to fill holes and merge close regions
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=2)

        # Find contours (moving regions)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes
        motion_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            motion_count += 1

            # draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(frame, f"motion #{motion_count} ({w}x{h})",
                        (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

        # FPS overlay
        if show_fps:
            frames += 1
            now = time.time()
            if now - prev_time >= 0.5:  # update every ~0.5s
                fps = frames / (now - prev_time)
                prev_time = now
                frames = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Detections: {motion_count}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Show result windows
        cv2.imshow("Motion Detection (Bounding Boxes)", frame)
        cv2.imshow("Mask (debug)", cleaned)

        # Press 'q' to quit, '+'/'-' to adjust sensitivity, '['/']' to change min_area
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            # more sensitive (lower var_threshold)
            var_threshold = max(2, var_threshold - 2)
            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=True
            )
            print(f"var_threshold -> {var_threshold} (more sensitive)")
        elif key == ord('-'):
            var_threshold = min(200, var_threshold + 2)
            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=True
            )
            print(f"var_threshold -> {var_threshold} (less sensitive)")
        elif key == ord('['):
            min_area = max(100, min_area - 200)
            print(f"min_area -> {min_area}")
        elif key == ord(']'):
            min_area += 200
            print(f"min_area -> {min_area}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change cam_index if you have multiple cameras (0,1,2…)
    main(cam_index=0)
