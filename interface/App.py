import streamlit as st
import numpy as np
from tools import MyPredictor
from PIL import Image

def load_settings() -> None:
    st.set_page_config(
        page_title="Medical Equipment Detection",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Title
    st.markdown("## Medical Personal Protective Equipment Detection")

def main() -> None:
    load_settings()
    uploaded_file = st.file_uploader(label="Choose a file", type=['png'])
    
    # Inference
    predictor = MyPredictor()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image_labelled = predictor.predict(image=image, iou_threshold=0.3)
        st.image(image_labelled)

if __name__=="__main__":
    main()