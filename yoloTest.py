from ultralytics import YOLO
import cv2

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # YOLO prediction
        results = model(frame, conf=0.5, verbose=False)

        # Extract predictions
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # BOUNDING BOX COORDINATES
                x1, y1, x2, y2 = box.xyxy[0]  # top-left and bottom-right

                # CLASS AND CONFIDENCE
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # Draw Rectangle
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Draw Text Label
                text = f"{label} {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Display annotated frame
        cv2.imshow("YOLOv8 Object Recognition", frame)

        # Quit on Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
