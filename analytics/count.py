from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Variables for counting

counted_ids = set()

car_count = 0
truck_count = 0
bus_count = 0
bike_count = 0

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
                if y1 < line_y < y2 and track_id not in counted_ids:
                    counted_ids.add(track_id)

                    if class_name == "car":
                        car_count += 1
                    elif class_name == "truck":
                        truck_count += 1
                    elif class_name == "bus":
                        bus_count += 1
                    elif class_name == "motorbike":
                        bike_count += 1

    # Display vehicle count
    count_text = f"Cars: {car_count}  Trucks: {truck_count}  Buses: {bus_count}  Bikes: {bike_count}"
    cv2.putText(frame,
                count_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)
    

    # Show output
    cv2.imshow("Vehicle Tracking and Counting", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()