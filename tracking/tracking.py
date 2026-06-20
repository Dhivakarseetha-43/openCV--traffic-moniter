from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track vehicles
    results = model.track(frame, persist=True)

    annotated_frame = results[0].plot()

    cv2.imshow("Vehicle Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()