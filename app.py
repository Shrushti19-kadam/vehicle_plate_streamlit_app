import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import easyocr

# Initialize OCR reader once
reader = easyocr.Reader(['en'])

# Store last detected text globally
last_text = ""

def ocr_on_frame(frame):
    global last_text
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    for detection in result:
        bbox, text, conf = detection
        if conf > 0.85 and text != last_text:
            last_text = text  # Update the last_text
            # Draw bounding box
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img

def video_frame_callback(frame):
    img = ocr_on_frame(frame)
    return img

st.title("Live Vehicle Plate Detection with Streamlit-WebRTC")

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

