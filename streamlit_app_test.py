from cv_pipeline import ssi_bounding_box
import streamlit as st
from pathlib import Path
import time

# Set page configuration
st.title("Smart Soccer Insights")
st.logo("02 Resources/logo.png", size="large")
st.header("By: :green[DGT-International]", divider='rainbow')
st.markdown(':green[DGT-International] developped a cutting-edge AI app that tracks soccer players.')

original_video = None

# sample video
use_sample = st.checkbox("Use a sample video", value=False)

sample_dir = Path('03 Data/sample_footage')
sample_files = list(sample_dir.glob('*'))

if use_sample:
    original_video = st.selectbox('Choose a sample video',
                                  sample_files,
                                  format_func=lambda x: x.name)

else:
    original_video = st.file_uploader(
        "Upload soccer footage to Smart Soccer Insights", type=["mp4", "mov", "avi"])

if original_video is not None:
    st.success("Original video uploaded successfully!")

# Display Original Video
if original_video is not None:
    show_bounding_boxes = st.checkbox("Show bounding boxes", value=False)
    if show_bounding_boxes:
        st.video("03 Data/121364_0_results_1_bounding_boxes.mp4")
    else:
        st.video(original_video)

# testing
#cacheing the function to avoid re-running it every time the app rerenders
@st.cache_data
def run_ssi_bounding_box(original_video, output_video):
    ssi_bounding_box(
        original_video,
        output_video
    )
    return output_video

if original_video is not None:
    st.header("Testing SSI Bounding Box Function")
    with st.spinner('Please allow about 15 minutes to detect and track players...', show_time=True):
        output_video = run_ssi_bounding_box(
            original_video,
            "03 Data/sample_footage/app_test_video.mp4"
        )

    st.success('Player detection and tracking complete!')
    
    if st.button('See results'):
        st.video("03 Data/sample_footage/app_test_video.mp4")