import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page settings
st.set_page_config(page_title="Face Detection App", layout="centered")

st.title("Face Detection using YOLOv8")

st.write("Upload an image and the model will detect faces.")

# Load trained model
model = YOLO("best.pt")

# Image uploader
uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    # Read image
    image = Image.open(uploaded)
    image_np = np.array(image)

    # Run detection
    results = model(image_np)

    # Draw bounding boxes
    annotated_image = results[0].plot()

    # Show result
    st.image(annotated_image, caption="Detected Faces", use_column_width=True)

    # Show detection details
    st.subheader("Detection Details")
    for i, box in enumerate(results[0].boxes):
        conf = box.conf[0].item()
        xyxy = box.xyxy[0].cpu().numpy()
        st.write(f"Face {i+1}")
        st.write(f"Confidence: {conf:.2f}")
        st.write(f"Bounding Box: {xyxy}")
