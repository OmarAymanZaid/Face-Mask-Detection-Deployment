import streamlit as st
import PIL.Image as Image
import requests
import io
import os

st.set_page_config(page_title="AI Mask Detector", page_icon="😷", layout="centered")

st.title("😷 Face Mask Detection System")
st.write("Upload a photo or use your webcam to check for mask compliance.")

# Read API URL from environment once — never from user input
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Optional: show it in sidebar as read-only info (no text_input)
st.sidebar.header("Settings")
st.sidebar.write(f"**API Endpoint:** `{API_URL}`")

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Webcam"])

def process_image(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    with st.spinner('Analyzing...'):
        try:
            files = {'file': ('image.jpg', byte_im, 'image/jpeg')}
            response = requests.post(API_URL, files=files)  # uses env var directly
            return response.json()
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
            return None

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        result = process_image(img)
        if result:
            st.success(f"Result: {result.get('class')}")
            st.write(result)

with tab2:
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        result = process_image(img)
        if result:
            st.success(f"Prediction: {result.get('class', 'Unknown')}")
            st.json(result)