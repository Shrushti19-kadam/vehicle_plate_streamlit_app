import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import easyocr
import time

# Title and intro
st.title("🚘 Live Number Plate Detection App")
st.write("🔍 Using EasyOCR + Streamlit-WebRTC")

# Initialize OCR reader once
reader = easyocr.Reader(['en'])

# Store last detected text and time globally
last_text = ""
last_detection_time = 0
TIMEOUT = 5  # seconds to wait before allowing same plate again

def ocr_on_frame(frame):
    global last_text, last_detection_time
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    current_time = time.time()
    for detection in result:
        bbox, text, conf = detection
        if conf > 0.85 and (text != last_text or current_time - last_detection_time > TIMEOUT):
            last_text = text
            last_detection_time = current_time

            # Draw bounding box
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return img

def video_frame_callback(frame):
    img = ocr_on_frame(frame)
    return img

# Start webcam with Streamlit-WebRTC
webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
