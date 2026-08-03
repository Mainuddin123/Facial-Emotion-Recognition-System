import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        return frame

st.title("Camera Test")

webrtc_streamer(
    key="test",
    video_processor_factory=VideoProcessor,
)