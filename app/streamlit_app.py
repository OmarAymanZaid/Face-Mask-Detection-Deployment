import streamlit as st
import PIL.Image as Image
import requests
import io
import os

st.set_page_config(page_title="AI Mask Detector", page_icon="😷", layout="centered")
st.title("😷 Face Mask Detection System")
st.write("Upload a photo or use your webcam to check for mask compliance.")

# Always read from environment — never from a mutable text input
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.sidebar.header("System Info")
st.sidebar.code(f"API: {API_URL}")

# Pull /health from the API to show live status in sidebar
try:
    health_url = API_URL.replace("/predict", "/health")
    h = requests.get(health_url, timeout=2).json()
    st.sidebar.success("API online" if h.get("status") == "ok" else "API degraded")
    st.sidebar.write(f"Device: `{h.get('device')}`")
    st.sidebar.write(f"AWS mode: `{h.get('aws_mode')}`")
except Exception:
    st.sidebar.error("API unreachable")

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Webcam"])


def process_image(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(
                API_URL,
                files={"file": ("image.jpg", byte_im, "image/jpeg")},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is the backend running?")
        except requests.exceptions.Timeout:
            st.error("Request timed out — model may still be loading.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    return None


def display_result(result: dict):
    status = result.get("status")
    cls    = result.get("class", "Unknown")
    conf   = result.get("confidence", 0)
    action = result.get("action", "")

    if status == "mask_on":
        st.success(f"✅ {cls} — {action}  ({conf:.1%} confidence)")
    else:
        st.error(f"🚫 {cls} — {action}  ({conf:.1%} confidence)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{conf:.1%}")
    col2.metric("Latency",    f"{result.get('latency_ms', 0):.0f} ms")
    col3.metric("Action",     action)

    with st.expander("Full response"):
        st.json(result)


with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_container_width=True)
        result = process_image(img)
        if result:
            display_result(result)

with tab2:
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        result = process_image(img)
        if result:
            display_result(result)