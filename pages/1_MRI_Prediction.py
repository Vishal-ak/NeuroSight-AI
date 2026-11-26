from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
from src.inference import load_predictor

st.set_page_config(page_title="MRI Prediction | NeuroSight AI",page_icon="🔬",layout="wide")
st.title("🔬 MRI Prediction")
st.caption("Upload a brain MRI image and run inference using a trained checkpoint.")
choices=sorted(list(Path("models").glob("**/*.pt"))+list(Path("models").glob("**/*.pth")))
if not choices:
    st.info("No trained checkpoint found. Train a model and place it under models/<model>/best_model.pt.")
    st.code("python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model cnn --epochs 15")
    st.stop()
selected=st.selectbox("Model checkpoint",choices,format_func=str)
upload=st.file_uploader("Upload MRI image",type=["jpg","jpeg","png","bmp"])
if upload:
    image=Image.open(upload).convert("RGB")
    left,right=st.columns(2)
    with left: st.image(image,caption="Uploaded MRI",use_container_width=True)
    try:
        @st.cache_resource(show_spinner="Loading model…")
        def predictor(path,mtime): return load_predictor(path)
        result=predictor(str(selected),selected.stat().st_mtime)(image)
        with right:
            st.subheader("Model output")
            st.metric("Predicted category",result["predicted_class"].replace("_"," ").title())
            st.caption("Highest model score; not a clinical probability.")
            scores=pd.DataFrame({"Class":list(result["probabilities"]),"Score":list(result["probabilities"].values())}).sort_values("Score",ascending=False)
            st.bar_chart(scores.set_index("Class"))
            st.dataframe(scores.style.format({"Score":"{:.2%}"}),use_container_width=True,hide_index=True)
    except Exception as exc: st.error(f"Inference failed: {exc}")
st.warning("Experimental image classification only. Not a medical diagnosis.")
