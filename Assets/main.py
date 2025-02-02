
from initialize import init_yolo
from video_processing import process_video

def main():
    # Initialize YOLO model
    model = init_yolo()
    
    # Start video processing
    # Use 0 for webcam or provide a video file path
    source="city vs madrid.mp4"
    process_video(model, source, conf_threshold=0.5)

if __name__ == "__main__":
    main()