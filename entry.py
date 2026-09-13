import cv2
import numpy as np
import csv
import time
from scipy.spatial import distance

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO dataset class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
video_path = "mallfoot.mp4"  # Change this to your CCTV video file
cap = cv2.VideoCapture(video_path)

# Get video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the vertical counting line (middle of the frame)
line_x = frame_width // 2  

# Initialize counters and tracking dictionary
entry_count = 0
exit_count = 0
tracked_objects = {}  # Dictionary to track detected people {id: (prev_x, prev_y)}
next_person_id = 0  # Unique ID for each person

# Create and open a CSV file for logging
log_file = "person_log.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Person_ID", "Action"])  # Header row

def log_entry(person_id, action):
    """Logs the entry or exit of a person in a CSV file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, person_id, action])
    print(f"Logged: {timestamp}, Person {person_id}, {action}")

def process_frame(frame):
    global entry_count, exit_count, tracked_objects, next_person_id
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    centroids = []

    # Detect persons
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] == "person":
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centroids.append((center_x, center_y))

    # Apply Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    # Dictionary for newly detected objects
    new_tracked_objects = {}

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x, center_y = centroids[i]

            # Match new detections with previously tracked objects
            matched_id = None
            for obj_id, (prev_x, prev_y) in tracked_objects.items():
                if distance.euclidean((center_x, center_y), (prev_x, prev_y)) < 50:
                    matched_id = obj_id
                    break

            # If no match, assign a new ID
            if matched_id is None:
                matched_id = next_person_id
                next_person_id += 1  # Increment ID counter

            # Check if the person crosses the vertical line
            if matched_id in tracked_objects:
                prev_x, prev_y = tracked_objects[matched_id]
                
                if prev_x < line_x and center_x >= line_x:
                    entry_count += 1  # Person entered (left to right)
                    print(f"Person Entered! Total Entry Count: {entry_count}")
                    log_entry(matched_id, "Entered")  # Log Entry

                elif prev_x > line_x and center_x <= line_x:
                    exit_count += 1  # Person exited (right to left)
                    print(f"Person Exited! Total Exit Count: {exit_count}")
                    log_entry(matched_id, "Exited")  # Log Exit

            # Update tracking info
            new_tracked_objects[matched_id] = (center_x, center_y)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Person {matched_id}', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update tracked objects with the new frame's detections
    tracked_objects = new_tracked_objects

    # Draw the vertical counting line
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 255, 0), 2)

    # Display Entry and Exit counts
    cv2.putText(frame, f'Entered: {entry_count}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exited: {exit_count}', (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = process_frame(frame)
    cv2.imshow("Person Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
