# ==========================================
# Import Libraries
# ==========================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from utils.mediapipe_helper import (
    extract_landmarks,
    get_face_bbox,
)

from utils.preprocessing import preprocess
from utils.predictor import predict_emotion

class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        bbox = get_face_bbox(img)

        if bbox is not None:

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            features = extract_landmarks(img)

            if features is not None:

                features = preprocess(features)

                emotion, confidence, _ = predict_emotion(features)

                cv2.putText(
                    img,
                    f"{emotion} ({confidence*100:.1f}%)",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Streamlit Configuration
# ==========================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown(
    """
<style>

.stApp{
    background:#0E1117;
}

.block-container{
    padding-top:2rem;
}

.main-title{
    text-align:center;
    font-size:52px;
    font-weight:bold;

    background:linear-gradient(
        90deg,
        red,
        orange,
        yellow,
        lime,
        cyan,
        blue,
        violet
    );

    background-size:400%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:20px;
    margin-bottom:25px;
}

.result-box{

    background:#1F2937;

    padding:25px;

    border-radius:15px;

    border:2px solid cyan;

    text-align:center;

}

.footer{

    text-align:center;

    color:white;

    margin-top:20px;

}

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# Header
# ==========================================

st.markdown(
    """
<h1 class="main-title">
😊 Facial Emotion Recognition
</h1>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<p class="sub-title">
MediaPipe Face Landmarker + Artificial Neural Network
</p>
""",
    unsafe_allow_html=True,
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.header("⚙ Settings")

input_mode = st.sidebar.radio(
    "Select Input",
    [
        "Upload Image",
        "Live Camera",
    ],
)

st.sidebar.divider()

st.sidebar.subheader("🤖 Model")

st.sidebar.success("Optuna Optimized ANN")

st.sidebar.info("MediaPipe Face Landmarker")

st.sidebar.divider()

st.sidebar.metric(
    label="Validation Accuracy",
    value="61.5%",
    delta="+2.2%",
)

st.sidebar.divider()

st.sidebar.subheader("😊 Emotion Classes")

for emotion in [
    "😠 Angry",
    "🤢 Disgust",
    "😨 Fear",
    "😊 Happy",
    "😐 Neutral",
    "😢 Sad",
    "😲 Surprise",
]:
    st.sidebar.write(emotion)

# ==========================================
# Input Section
# ==========================================

image = None

if input_mode == "Upload Image":

    st.subheader("📤 Upload Face Image")

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image = np.array(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
else:

    st.subheader("🎥 Live Camera")

    webrtc_streamer(
        key="emotion-recognition",

        video_processor_factory=EmotionProcessor,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.relay.metered.ca:80"],
                },
                {
                    "urls": ["turn:global.relay.metered.ca:80"],
                    "username": "d4a26fc7710d51300f0fc52d",
                    "credential": "cibLU/J0Zvz5wiVZ",
                },
                {
                    "urls": ["turn:global.relay.metered.ca:80?transport=tcp"],
                    "username": "d4a26fc7710d51300f0fc52d",
                    "credential": "cibLU/J0Zvz5wiVZ",
                },
                {
                    "urls": ["turn:global.relay.metered.ca:443"],
                    "username": "d4a26fc7710d51300f0fc52d",
                    "credential": "cibLU/J0Zvz5wiVZ",
                },
                {
                    "urls": ["turns:global.relay.metered.ca:443?transport=tcp"],
                    "username": "d4a26fc7710d51300f0fc52d",
                    "credential": "cibLU/J0Zvz5wiVZ",
                },
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )

    image = None


# ==========================================
# Image Prediction
# ==========================================

if image is not None:

    col1, col2 = st.columns([2, 1])

    # --------------------------------------
    # Display Uploaded Image
    # --------------------------------------

    with col1:
        if input_mode == "Upload Image":
            st.subheader("📷 Input Image")
        else:
            st.subheader("🎥 Live Camera")

        preview = image.copy()

        bbox = get_face_bbox(preview)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                preview,(x1, y1),(x2, y2),
                (0, 255, 0),2)
        st.image(
            cv2.cvtColor(
            preview,
            cv2.COLOR_BGR2RGB),
        use_container_width=True)
    

    # --------------------------------------
    # Landmark Extraction
    # --------------------------------------

    features = extract_landmarks(image)

    with col2:

        st.subheader("🤖 Prediction")

        if features is None:

            st.error(
                "❌ No face detected. Please upload a clear frontal face image."
            )

        else:

            # Preprocess
            features = preprocess(features)

            # Predict
            emotion, confidence, _ = predict_emotion(features)

            # Result Card

            st.markdown(
                f"""
                <div class="result-box">

                <h2 style="color:#00E5FF;">
                Predicted Emotion
                </h2>

                <h1 style="color:white;">
                {emotion}
                </h1>

                <h3 style="color:#00FF7F;">
                Confidence : {confidence*100:.2f}%
                </h3>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.success("Prediction Completed Successfully ✔")

            st.info(f"Detected Emotion : {emotion}")

            # --------------------------------------
            # Download Report
            # --------------------------------------

            report = f"""
==========================================
Facial Emotion Recognition Result
==========================================

Predicted Emotion : {emotion}

Confidence : {confidence*100:.2f} %

Model :
Optuna Optimized ANN

Feature Extractor :
MediaPipe Face Landmarker

Features :
1434

Generated using Streamlit
"""

            st.download_button(
                label="📄 Download Prediction Report",
                data=report,
                file_name="emotion_prediction.txt",
                mime="text/plain",
            )

# ==========================================
# Footer
# ==========================================

st.divider()

st.markdown(
    """
<div class="footer">

<h2 style="color:#00E5FF;">
😊 Facial Emotion Recognition System
</h2>

<p style="color:white;font-size:16px;">

Developed using

<strong>MediaPipe Face Landmarker</strong> •

<strong>TensorFlow / Keras</strong> •

<strong>Streamlit</strong>

</p>

<p style="color:#FFD700;font-size:18px;">

Developed by
<strong>Khaja Mainuddin</strong>

</p>

</div>
""",
    unsafe_allow_html=True,
)


