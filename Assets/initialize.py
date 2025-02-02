from ultralytics import YOLO

def init_yolo():
    """Initialize YOLO model"""
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model
    return model