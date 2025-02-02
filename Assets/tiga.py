import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os

# Load YOLOv8 model
model = YOLO('yolo11n.pt')
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

def process_video(video_path):
    """Process video, track players, and return processed video path."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create temporary output video file
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, "processed_video.mp4")

    # Define codec and create VideoWriter
    if os.name == 'nt':  # Windows
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:  # Linux/Mac
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    player_positions = {}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Streamlit progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Detect and track objects
        results = model.track(frame, persist=True)

        # Draw bounding boxes and collect player positions
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    tracker_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Store normalized center position
                    center_x, center_y = get_center_position(x1, y1, x2, y2)
                    normalized_x = center_x / frame_width
                    normalized_y = center_y / frame_height

                    if tracker_id not in player_positions:
                        player_positions[tracker_id] = []

                    player_positions[tracker_id].append({'x': normalized_x, 'y': normalized_y})

                    # Draw bounding box and ID label
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

    status_text.text("Video processing complete!")
    return player_positions, temp_output_path

def main():
    st.title("Player Tracking and Position Analysis")

    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    if video_file:
        # Create temporary input file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(video_file.read())
        temp_input_path = temp_input.name
        temp_input.close()

        if st.button("Process Video"):
            # Process video and get player positions and output path
            player_positions, processed_video_path = process_video(temp_input_path)

            # Display processed video
            st.header("Processed Video with Player Tracking")
            with open(processed_video_path, 'rb') as f:
                st.video(f.read())

            # Clean up temporary files
            os.remove(temp_input_path)
            os.remove(processed_video_path)

            # Store player positions in session state
            st.session_state.player_positions = player_positions

    # Check if player positions are available
    if 'player_positions' in st.session_state:
        player_positions = st.session_state.player_positions

        # Display player heatmap buttons
        st.header("Select Player to View Heatmap")
        selected_player = st.session_state.get('selected_player')

        # Create buttons for each player
        for player_id in player_positions:
            if st.button(f"Show Heatmap for Player {player_id}", key=f"player_{player_id}"):
                st.session_state.selected_player = player_id  # Store selected player

        # Display the selected player's heatmap
        if selected_player is not None:
            st.subheader(f"Player {selected_player} Heatmap")
            fig = create_heatmap(player_positions[selected_player], selected_player)
            if fig:
                st.pyplot(fig)

            # Provide option to download player data
            df = pd.DataFrame(player_positions[selected_player])
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"Download Player {selected_player} Data",
                data=csv,
                file_name=f'player_{selected_player}_positions.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()