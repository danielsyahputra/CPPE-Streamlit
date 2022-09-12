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
    sidebar = st.sidebar

def main() -> None:
    load_settings()
    uploaded_file = st.file_uploader(label="Choose a file", type=['png'])
    
    # Inference
    predictor = MyPredictor()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        start_time = timer()
        image_labelled = predictor.predict(image=image, iou_threshold=0.5)
        end_time = timer()
        st.write(f"Total time: {end_time - start_time:.3f} seconds")
        st.image(image_labelled)

if __name__=="__main__":
    main()