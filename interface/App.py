import streamlit as st
import numpy as np
from tools import MyPredictor
from PIL import Image
from timeit import default_timer as timer

def load_settings() -> None:
    st.set_page_config(
        page_title="Medical Equipment Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.markdown("## Medical Personal Protective Equipment Detection")

    global sidebar
    global iou_threshold
    global probability_threshold
    global model_option

    sidebar = st.sidebar

    with open("test/test_images.zip", "rb") as fp:
        btn = sidebar.download_button(
            label="Download Test Images",
            data=fp,
            file_name="test_images.zip",
            mime="application/zip"
        )

    sidebar.write("Try to tune this value for better post-processing result.")
    iou_threshold = sidebar.number_input("IoU Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    probability_threshold = sidebar.number_input("Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    model_option = sidebar.selectbox(label="Choose the baseline model", options=["SSD", "FasterRCNN"])

def main() -> None:
    load_settings()
    uploaded_file = st.file_uploader(label="Choose a file", type=['png'])
    
    # Inference
    predictor = MyPredictor(baseline_model=model_option)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        start_time = timer()
        image_labelled = predictor.predict(image=image,probability_threshold=probability_threshold, iou_threshold=iou_threshold)
        end_time = timer()
        st.write(f"Total time: {end_time - start_time:.3f} seconds")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="text-align: center;">Before Image</p><hr>', unsafe_allow_html=True)
            st.image(image)
        with col2:
            st.markdown('<p style="text-align: center;">After Image</p><hr>', unsafe_allow_html=True)
            st.image(image_labelled)

if __name__=="__main__":
    main()