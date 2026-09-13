import urllib.request
import os

# URLs for YOLOv4-tiny files
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"

# Save paths
cfg_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"

def download_file(url, save_path):
    """Download file if it doesn't exist."""
    if not os.path.exists(save_path):
        print(f"Downloading {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded: {save_path}")
    else:
        print(f"{save_path} already exists.")

# Download YOLOv4-tiny files
download_file(cfg_url, cfg_path)
download_file(weights_url, weights_path)

print("All files are ready!")
