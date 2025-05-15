import streamlit as st
import os
import gdown

st.title("🔥 Fire Detection App (Test Deployment)")

model_path = "best.pt"

if not os.path.exists(model_path):
    st.info("Downloading model file...")
    url = "https://drive.google.com/uc?id=1INHMFa1yb3reB6ezi7nJvjGOVEyvvIAm"
    gdown.download(url, model_path, quiet=False)
    st.success("Model downloaded!")

try:
    from ultralytics import YOLO

    model = YOLO(model_path)
    st.success("Model loaded successfully!")

    st.markdown("You can now upload images/videos to detect fire (build your full logic here).")

except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
