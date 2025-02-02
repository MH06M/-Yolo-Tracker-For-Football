import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os
from collections import deque

# Load YOLOv8 model
model = YOLO('yolo11n.pt')

# Advanced configuration options
CONFIG = {
    'confidence_threshold': 0.5,        # Minimum detection confidence
    'iou_threshold': 0.3,              # IOU threshold for NMS
    'track_buffer': 30,                # Frames to keep track of lost objects
    'min_box_area': 100,               # Minimum bounding box area
    'motion_threshold': 50,            # Maximum allowed motion between frames
    'track_history_length': 30,        # Number of frames for position history
    'frame_skip': 1,                   # Process every nth frame (1 = no skip)
}

class EnhancedTracker:
    def __init__(self):
        self.track_history = {}  # Store position history for each tracked object
        self.lost_tracks = {}    # Store temporarily lost tracks
        self.kalman_filters = {} # Store Kalman filters for each track
        
    def init_kalman_filter(self):
        """Initialize Kalman filter for position tracking."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                      [0, 1, 0, 1],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32) * 0.03
        return kf

    def update_track(self, track_id, position):
        """Update track with new position using Kalman filter."""
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self.init_kalman_filter()
            self.track_history[track_id] = deque(maxlen=CONFIG['track_history_length'])

        kf = self.kalman_filters[track_id]
        
        # Predict
        prediction = kf.predict()
        
        # Update with measurement
        measurement = np.array([[position[0]], [position[1]]], dtype=np.float32)
        kf.correct(measurement)
        
        # Store position history
        self.track_history[track_id].append(position)
        
        return prediction[:2].flatten()

    def get_track_history(self, track_id):
        """Get position history for a track."""
        return list(self.track_history.get(track_id, []))


def preprocess_frame(frame):
    """Enhance frame for better detection."""
    # Convert to float32
    frame_float = frame.astype(np.float32) / 255.0
    
    # Apply contrast enhancement
    alpha = 1.5  # Contrast control
    beta = 0.0   # Brightness control
    frame_enhanced = cv2.convertScaleAbs(frame_float, alpha=alpha, beta=beta)
    
    # Apply denoising
    frame_denoised = cv2.fastNlMeansDenoisingColored(frame_enhanced, None, 10, 10, 7, 21)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    frame_sharp = cv2.filter2D(frame_denoised, -1, kernel)
    
    return frame_sharp


def get_center_position(x1, y1, x2, y2):
    """Calculate center position of bounding box."""
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def create_heatmap(player_positions, player_id):
    """Create a heatmap for a specific player using seaborn with a pitch background."""
    if not player_positions:
        st.warning(f"No position data available for player {player_id}")
        return

    # Convert positions to DataFrame
    df = pd.DataFrame(player_positions)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load and plot the pitch background image
    pitch_image = plt.imread('pitch.jpg')
    ax.imshow(pitch_image, extent=(0, 1, 0, 1), alpha=0.8, aspect='auto')

    # Create heatmap using seaborn
    sns.kdeplot(data=df, x='x', y='y', cmap='YlOrRd', fill=True, ax=ax)

    # Set plot limits to match normalized coordinates
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Invert y-axis to match video coordinate system
    ax.invert_yaxis()

    # Set title and labels
    plt.title(f'Player {player_id} Position Heatmap')
    plt.xlabel('X Position (normalized)')
    plt.ylabel('Y Position (normalized)')

    return fig

def process_video(video_path, specific_player_id=None):
    """Process video with enhanced tracking."""
    cap = cv2.VideoCapture(video_path)
    
    # Initialize model with custom parameters
    model = YOLO('yolo11n.pt')
    model.conf = CONFIG['confidence_threshold']
    model.iou = CONFIG['iou_threshold']
    
    # Initialize enhanced tracker
    tracker = EnhancedTracker()

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, "processed_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'H264') if os.name == 'nt' else cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    player_positions = {}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Streamlit progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames if configured
        if frame_count % CONFIG['frame_skip'] != 0:
            continue
            
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Run detection and tracking
        results = model.track(processed_frame, persist=True, verbose=False)
        
        # Create display frame
        display_frame = frame.copy()

        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                if box.id is None:
                    continue
                    
                tracker_id = int(box.id[0])
                
                # Skip if tracking specific player
                if specific_player_id is not None and tracker_id != specific_player_id:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Filter small boxes
                if (x2 - x1) * (y2 - y1) < CONFIG['min_box_area']:
                    continue

                # Get center position
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Update position with Kalman filter
                predicted_pos = tracker.update_track(tracker_id, (center_x, center_y))
                
                # Store normalized position
                normalized_x = center_x / frame_width
                normalized_y = center_y / frame_height
                
                if tracker_id not in player_positions:
                    player_positions[tracker_id] = []
                
                player_positions[tracker_id].append({
                    'x': normalized_x,
                    'y': normalized_y,
                    'frame': frame_count
                })

                # Draw tracking visualization
                color = (0, 0, 255) if specific_player_id and tracker_id == specific_player_id else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"ID: {tracker_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Draw trail
                track_history = tracker.get_track_history(tracker_id)
                if len(track_history) > 1:
                    for i in range(len(track_history) - 1):
                        start_pos = track_history[i]
                        end_pos = track_history[i + 1]
                        alpha = (i + 1) / len(track_history)
                        trail_color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                        cv2.line(display_frame, 
                                (int(start_pos[0]), int(start_pos[1])),
                                (int(end_pos[0]), int(end_pos[1])),
                                trail_color, 2)

        out.write(display_frame)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    status_text.text("Video processing complete!")
    return player_positions, temp_output_path

def main():
    st.title("Player Tracking and Position Analysis")

    # Store the uploaded video path in session state
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None

    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    if video_file:
        # Create temporary input file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(video_file.read())
        st.session_state.video_path = temp_input.name
        temp_input.close()

        if st.button("Process Video"):
            # Process video and get player positions and output path
            player_positions, processed_video_path = process_video(st.session_state.video_path)

            # Display processed video
            st.header("Processed Video with Player Tracking")
            with open(processed_video_path, 'rb') as f:
                st.video(f.read())

            # Clean up temporary processed video
            os.remove(processed_video_path)

            # Store player positions in session state
            st.session_state.player_positions = player_positions

    # Check if player positions are available
    if 'player_positions' in st.session_state:
        player_positions = st.session_state.player_positions

        # Display player heatmap buttons
        st.header("Select Player to View Heatmap and Track")
        
        # Create columns for a better layout
        cols = st.columns(3)
        
        # Create buttons for each player
        for idx, player_id in enumerate(player_positions):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.button(f"Track Player {player_id}", key=f"player_{player_id}"):
                    st.session_state.selected_player = player_id
                    
                    if st.session_state.video_path:
                        # Process video again for specific player
                        st.subheader(f"Tracking Player {player_id}")
                        player_positions, processed_video_path = process_video(
                            st.session_state.video_path, 
                            specific_player_id=player_id
                        )
                        
                        # Display processed video
                        with open(processed_video_path, 'rb') as f:
                            st.video(f.read())
                        
                        # Clean up temporary processed video
                        os.remove(processed_video_path)
                        
                        # Display heatmap
                        st.subheader(f"Player {player_id} Heatmap")
                        fig = create_heatmap(player_positions[player_id], player_id)
                        if fig:
                            st.pyplot(fig)

                        # Provide option to download player data
                        df = pd.DataFrame(player_positions[player_id])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"Download Player {player_id} Data",
                            data=csv,
                            file_name=f'player_{player_id}_positions.csv',
                            mime='text/csv'
                        )

    # Clean up temporary input file when the app is done
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        try:
            os.remove(st.session_state.video_path)
        except:
            pass

if __name__ == "__main__":
    main()