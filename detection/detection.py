from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture("videos/traffic.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()