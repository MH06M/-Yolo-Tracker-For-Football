import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Load YOLO model
model = YOLO('yolo11n-seg.pt')

def process_video(video_path):
    """Process video, track players, and return processed video as bytes."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video = "processed_video.mp4"
    out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_width, frame_height))

    player_positions = {}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Streamlit progress and status
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Detect and track objects
        results = model.track(frame, persist=True)

        # Draw bounding boxes and track positions
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    tracker_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Calculate and store normalized positions
                    center_x, center_y = get_center_position(x1, y1, x2, y2)
                    normalized_x = center_x / frame_width
                    normalized_y = center_y / frame_height

                    if tracker_id not in player_positions:
                        player_positions[tracker_id] = []

                    player_positions[tracker_id].append({'x': normalized_x, 'y': normalized_y})

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"ID: {tracker_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                    )

        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Read the processed video as bytes
    with open(temp_video, "rb") as f:
        video_bytes = f.read()

    status_text.text("Video processing complete!")
    return player_positions, video_bytes

def get_center_position(x1, y1, x2, y2):
    """Calculate center position of bounding box."""
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def main():
    st.title("Player Tracking and Position Analysis")

    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())

        if st.button("Process Video"):
            player_positions, video_bytes = process_video("temp_video.mp4")

            # Display processed video
            st.header("Processed Video with Player Tracking")
            st.video(video_bytes)

            # Display heatmaps
            st.header("Player Heatmaps")
            cols = st.columns(2)

            for idx, (player_id, positions) in enumerate(player_positions.items()):
                with cols[idx % 2]:
                    st.subheader(f"Player {player_id}")
                    fig = create_heatmap(positions, player_id, 1, 1)  # Normalized axes
                    if fig:
                        st.pyplot(fig)

                    df = pd.DataFrame(positions)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download Player {player_id} Data",
                        data=csv,
                        file_name=f'player_{player_id}_positions.csv',
                        mime='text/csv',
                        key=f"download_{player_id}"
                    )

            # Display tracking statistics
            st.header("Tracking Statistics")
            for player_id, positions in player_positions.items():
                st.text(f"Player {player_id}: {len(positions)} positions tracked")

if __name__ == "__main__":
    main()
