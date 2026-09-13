import cv2
import numpy as np
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

# Initialize counter and tracking dictionary
person_count = 0
tracked_centroids = []  # Store centroids of detected people

def process_frame(frame):
    global person_count, tracked_centroids
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    new_centroids = []

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
                new_centroids.append((center_x, center_y))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x, center_y = new_centroids[i]

            # Check if this person is new
            if not any(distance.euclidean((center_x, center_y), prev) < 50 for prev in tracked_centroids):
                person_count += 1
                tracked_centroids.append((center_x, center_y))  # Track this new person
                print(f"New Person Detected! Total Count: {person_count}")

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(frame, f'Total Count: {person_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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