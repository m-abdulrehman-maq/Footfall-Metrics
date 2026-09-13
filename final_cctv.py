import gradio as gr
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

# Initialize counters and tracking dictionary
entry_count = 0
exit_count = 0
tracked_objects = {}
next_person_id = 0
log_file = "person_log.csv"

# Function to log entry/exit
def log_entry(person_id, action):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, person_id, action])
    print(f"Logged: {timestamp}, Person {person_id}, {action}")

# Function to process video and count persons
def process_video(video_path):
    global entry_count, exit_count, tracked_objects, next_person_id
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    line_x = frame_width // 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(out_layers)
        
        boxes, centroids = [], []
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
                    centroids.append((center_x, center_y))

        indexes = cv2.dnn.NMSBoxes(boxes, [0.5] * len(boxes), 0.4, 0.3)
        new_tracked_objects = {}

        for i in indexes:
            x, y, w, h = boxes[i]
            center_x, center_y = centroids[i]
            matched_id = None

            for obj_id, (prev_x, prev_y) in tracked_objects.items():
                if distance.euclidean((center_x, center_y), (prev_x, prev_y)) < 50:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = next_person_id
                next_person_id += 1

            if matched_id in tracked_objects:
                prev_x, _ = tracked_objects[matched_id]
                if prev_x < line_x and center_x >= line_x:
                    entry_count += 1
                    log_entry(matched_id, "Entered")
                elif prev_x > line_x and center_x <= line_x:
                    exit_count += 1
                    log_entry(matched_id, "Exited")

            new_tracked_objects[matched_id] = (center_x, center_y)

        tracked_objects = new_tracked_objects
    
    cap.release()
    return entry_count, exit_count

# Gradio UI
def analyze_video(video):
    entered, exited = process_video(video)
    return f"Entered: {entered}, Exited: {exited}"

with gr.Blocks() as demo:
    video_input = gr.Video(label="Upload CCTV Footage")
    output_text = gr.Textbox(label="Entry/Exit Count")
    analyze_button = gr.Button("Analyze Video")
    analyze_button.click(analyze_video, inputs=video_input, outputs=output_text)

demo.launch(show_error=True)
