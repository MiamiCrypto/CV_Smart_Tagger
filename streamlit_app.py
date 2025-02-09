import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
import io

# Fix OpenCV import issue
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")

# Streamlit App Title
st.title("Smart Tagger: AI-Powered Image Annotation Tool")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert Image to OpenCV Format
    image_cv = np.array(image.convert("RGB"))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Run YOLO Object Detection
    results = yolo_model(image_cv)
    
    # Extract Annotations
    annotations = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls.item())]  # Convert class index to label
            annotations.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "label": label})
    
    # Convert to DataFrame
    df = pd.DataFrame(annotations)
    st.dataframe(df)
    
    # Overlay Bounding Boxes on Image
    for ann in annotations:
        cv2.rectangle(image_cv, (ann["x"], ann["y"]), (ann["x"] + ann["width"], ann["y"] + ann["height"]), (0, 255, 0), 2)
        label = f"{ann['label']}"
        cv2.putText(image_cv, label, (ann["x"], ann["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display Annotated Image
    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Auto-Annotated Image", use_container_width=True)
    
    # Download Option
    csv = df.to_csv(index=False)
    st.download_button("Download Annotations", csv, "annotations.csv", "text/csv", key="download-csv")
