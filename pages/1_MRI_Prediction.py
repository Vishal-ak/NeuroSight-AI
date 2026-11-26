from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
from src.inference import load_predictor

st.set_page_config(page_title="MRI Prediction | NeuroSight AI",page_icon="🔬",layout="wide")
st.title("🔬 MRI Prediction")
st.caption("Upload a brain MRI image and run an individual model or the selected overall-best system.")
choices=sorted(list(Path("models").glob("**/*.pt"))+list(Path("models").glob("**/*.pth")))
if not choices:
    st.info("No trained checkpoint found. Train the three classifiers and run ensemble selection first.")
    st.code("python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model cnn --epochs 15")
    st.stop()
overall=Path("models/overall_best/best_model.pt")
default_index=choices.index(overall) if overall in choices else 0
selected=st.selectbox("Model checkpoint",choices,index=default_index,format_func=lambda p: "Overall best — "+str(p) if p==overall else str(p))
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
            st.caption("Model score; not a calibrated clinical probability.")
            st.caption("Model: "+result.get("model_used",selected.parent.name))
            scores=pd.DataFrame({"Class":list(result["probabilities"]),"Score":list(result["probabilities"].values())}).sort_values("Score",ascending=False)
            st.bar_chart(scores.set_index("Class"))
            st.dataframe(scores.style.format({"Score":"{:.2%}"}),use_container_width=True,hide_index=True)
    except Exception as exc: st.error(f"Inference failed: {exc}")
st.warning("Experimental image classification only. Not a medical diagnosis.")
