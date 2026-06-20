from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Variables for counting
counted_ids = set()
vehicle_count = 0

# Position of counting line
line_y = 300

while True:
    ret, frame = cap.read()

    # Stop when video ends
    if not ret:
        break

    # Track objects
    results = model.track(frame, persist=True)

    # Draw counting line
    cv2.line(frame,
             (0, line_y),
             (frame.shape[1], line_y),
             (0, 0, 255),
             2)

    # Process detections
    for r in results:
        boxes = r.boxes

        for box in boxes:

            # Check if tracking ID exists
            if box.id is not None:

                # Get tracking ID and class
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              2)

                # Small label
                label = f"{class_name} ID:{track_id}"

                cv2.putText(frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,      # Smaller font
                            (0, 255, 0),
                            1)

                # Count vehicle when crossing line
                center_y = (y1 + y2) // 2

                if center_y > line_y and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_count += 1

    # Display vehicle count
    cv2.putText(frame,
                f"Count: {vehicle_count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    # Show output
    cv2.imshow("Vehicle Tracking and Counting", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()