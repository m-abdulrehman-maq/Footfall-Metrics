# This is a Gradio app that counts the number of persons appearing in a video stream from a CCTV footage.
import gradio as gr
import numpy as np
import cv2

# Function to count persons in a frame
def count_persons(video):
    # Initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Read the video frame by frame
    cap = cv2.VideoCapture(video)
    person_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons in the frame
        boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Count the number of persons detected
        person_count += len(boxes)

    cap.release()
    return person_count

# Create a Gradio interface
with gr.Blocks() as demo:
    # Input component for the video
    video_input = gr.Video(label="Upload CCTV Footage")

    # Output component for the person count
    person_count_output = gr.Number(label="Number of Persons Detected")

    # Button to trigger the person counting function
    count_button = gr.Button("Count Persons")

    # Event listener to trigger the function when the button is clicked
    count_button.click(count_persons, inputs=video_input, outputs=person_count_output)

# Launch the interface
if __name__ == "__main__":
    demo.launch(show_error=True)