import streamlit as st
import tools

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

if __name__=="__main__":
    main()