import streamlit as st
from pathlib import Path

# Set page configuration
st.title("Smart Soccer Insights")
st.header("By: :green[DGT-International]", divider='rainbow')
st.markdown(':green[DGT-International] developped a cutting-edge AI app that tracks soccer players.')

original_video = None

# sample video
use_sample = st.checkbox("Use a sample video")

sample_dir = Path('03 Data/sample_footage')
sample_files = list(sample_dir.glob('*'))

if use_sample:
    selected = st.selectbox('Choose a sample video',
                            sample_files,
                            format_func=lambda x: x.name)
    
    st.video(selected)



#     selected = st.checkbox('Choose a sample video', sample_files)

#     original_video = open(selected, 'rb')

# else:
#     original_video = None
#     st.write("Please upload your own soccer footage.")
    
# # Original Video
# if original_video is None:
#     original_video = st.file_uploader("Upload soccer footage to Smart Soccer Insights", type=["mp4", "mov", "avi"])
#     st.success("Original video uploaded successfully!")

#st.video(original_video, format="video/mp4", start_time=0)
    
