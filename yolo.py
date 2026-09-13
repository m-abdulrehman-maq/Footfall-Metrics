import urllib.request

# YOLOv4 files download URLs
weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
coco_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Download and save the files
urllib.request.urlretrieve(weights_url, "yolov4.weights")
urllib.request.urlretrieve(cfg_url, "yolov4.cfg")
urllib.request.urlretrieve(coco_url, "coco.names")

print("YOLOv4 files downloaded successfully!")
