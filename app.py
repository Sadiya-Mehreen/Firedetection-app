import streamlit as st
import os
import gdown

st.title("🔥 Fire Detection App (Test Deployment)")

# Model file name
model_path = "best.pt"

# Download model if not exists
if not os.path.exists(model_path):
    st.info("Downloading model file...")
    # Replace this with your Google Drive file ID for best.pt
    url = "https://drive.google.com/file/d/1INHMFa1yb3reB6ezi7nJvjGOVEyvvIAm/view?usp=drive_link"
    gdown.download(url, model_path, quiet=False)
    st.success("Model downloaded!")

# Placeholder for YOLO import and prediction
try:
    from ultralytics import YOLO

    model = YOLO(model_path)
    st.success("Model loaded successfully!")

    st.markdown("You can now upload images/videos to detect fire (build your full logic here).")

except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

