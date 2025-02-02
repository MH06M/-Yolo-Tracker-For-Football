import cv2
import time

def process_video(model, source=0, conf_threshold=0.5):
    """
    Process video stream with YOLO detection
    
    Args:
        model: YOLO model instance
        source: Video source (0 for webcam, or video file path)
        conf_threshold: Confidence threshold for detections
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize FPS counter
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detectionq
            # results = model(frame, conf=conf_threshold, stream=True)
            results = model.track(frame, persist=True)
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f'{model.names[cls]}: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('YOLO Detection', frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
