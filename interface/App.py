import streamlit as st
import cv2
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
        boxes, labels, scores = predictor.predict(image=image, iou_threshold=0.3)
        for i, box in enumerate(boxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            color = predictor.colors[labels[i]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, predictor.label[labels[i]], (xmin, ymax - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        st.image(image)

if __name__=="__main__":
    main()