import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Title of the app
st.title("Violence Detection Training and Visualization")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an option:", ["Home", "Upload Video", "Training Graphs"])

# Home Page
if options == "Home":
    st.header("Welcome to the Violence Detection App")
    st.write("""
        This app allows you to:
        - Upload videos for preprocessing.
        - Visualize training and validation graphs.
        - Interact with the violence detection model.
    """)

# Upload Video Page
elif options == "Upload Video":
    st.header("Upload a Video for Preprocessing")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.video(temp_file_path)
        st.write("Video uploaded successfully!")
        
        # Preprocess the video
        def extract_frames(video_path, target_height, target_width, num_frames):
            frames = []
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (target_width, target_height))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()
            return np.array(frames)
        
        target_height, target_width, num_frames = 120, 120, 16
        frames = extract_frames(temp_file_path, target_height, target_width, num_frames)
        
        if len(frames) == num_frames:
            st.write("Frames extracted successfully!")
            st.image(frames, caption="Extracted Frames", use_column_width=True)
        else:
            st.write("Failed to extract sufficient frames from the video.")

# Training Graphs Page
elif options == "Training Graphs":
    st.header("Training and Validation Graphs")
    
    # Example training history data
    history_data = {
        "loss": [0.69, 0.59, 0.48, 0.39, 0.33],
        "val_loss": [0.68, 0.58, 0.50, 0.42, 0.38],
        "accuracy": [0.50, 0.66, 0.77, 0.85, 0.90],
        "val_accuracy": [0.52, 0.65, 0.75, 0.83, 0.88]
    }
    
    # Generate epochs
    epochs = range(1, len(history_data["loss"]) + 1)
    
    # Plot for loss
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, history_data["loss"], 'bo-', label='Training Loss')
    ax[0].plot(epochs, history_data["val_loss"], 'ro-', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Plot for accuracy
    ax[1].plot(epochs, history_data["accuracy"], 'bo-', label='Training Accuracy')
    ax[1].plot(epochs, history_data["val_accuracy"], 'ro-', label='Validation Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    st.pyplot(fig)
    
    # Summary of training
    final_accuracy = history_data["accuracy"][-1]
    final_val_accuracy = history_data["val_accuracy"][-1]
    final_loss = history_data["loss"][-1]
    final_val_loss = history_data["val_loss"][-1]
    
    st.write("### Training Summary")
    st.write(f"**Final Training Accuracy:** {final_accuracy * 100:.2f}%")
    st.write(f"**Final Validation Accuracy:** {final_val_accuracy * 100:.2f}%")
    st.write(f"**Final Training Loss:** {final_loss:.4f}")
    st.write(f"**Final Validation Loss:** {final_val_loss:.4f}")