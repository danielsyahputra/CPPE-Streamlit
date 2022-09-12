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
    global is_json_output

    sidebar = st.sidebar
    sidebar.write("Try to tune this value for better post-processing result.")
    iou_threshold = sidebar.number_input("IoU Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    probability_threshold = sidebar.number_input("Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    is_json_output = sidebar.checkbox(label="JSON Output", value=False)

def main() -> None:
    load_settings()
    uploaded_file = st.file_uploader(label="Choose a file", type=['png'])
    
    # Inference
    predictor = MyPredictor()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        start_time = timer()
        image_labelled, preds = predictor.predict(image=image,probability_threshold=probability_threshold, iou_threshold=iou_threshold)
        end_time = timer()
        st.write(f"Total time: {end_time - start_time:.3f} seconds")

        if is_json_output:
            st.json(preds)
        else:
            st.image(image_labelled)

if __name__=="__main__":
    main()