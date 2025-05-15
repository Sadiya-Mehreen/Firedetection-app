import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import shutil
import glob

# Load your trained YOLO model
model = YOLO("best.pt")

# Streamlit page settings
st.set_page_config(page_title="🔥 Fire Detection", layout="centered")
st.title("🔥 Fire Detection using YOLOv8")
st.markdown("Upload an **image or video** to detect fire with your custom-trained YOLO model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Save uploaded file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_type.startswith("image"):
        try:
            # Display uploaded image
            image = Image.open(temp_path)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Run YOLO inference
            results = model.predict(source=temp_path, imgsz=640, conf=0.5, save=True)

            # Locate the predicted image (glob to find any image file in output directory)
            result_dir = results[0].save_dir
            pred_images = glob.glob(os.path.join(result_dir, "*.jpg")) + glob.glob(os.path.join(result_dir, "*.png"))

            if pred_images:
                st.image(pred_images[0], caption="Detection Result", use_container_width=True)
                with open(pred_images[0], "rb") as f:
                    st.download_button("Download Result", f, file_name="result.jpg")
            else:
                st.warning("No result image found.")

        except Exception as e:
            st.error(f"Could not open image: {e}")

    elif file_type.startswith("video"):
        st.video(temp_path)

        with st.spinner("Processing video..."):
            results = model.predict(source=temp_path, imgsz=640, conf=0.5, save=True)

        result_dir = results[0].save_dir
        pred_videos = glob.glob(os.path.join(result_dir, "*.mp4")) + glob.glob(os.path.join(result_dir, "*.avi"))

        if pred_videos:
            st.success("Video processed successfully!")
            st.video(pred_videos[0])
            with open(pred_videos[0], "rb") as f:
                st.download_button("Download Result Video", f, file_name="result_video.mp4")
        else:
            st.warning("No result video found.")

    # Clean up
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        st.warning(f"Temp cleanup error: {e}")

