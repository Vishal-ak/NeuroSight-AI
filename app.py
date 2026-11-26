import streamlit as st

st.set_page_config(page_title="NeuroSight AI", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1400px}
[data-testid="stMetric"]{background:rgba(120,150,170,.08);border:1px solid rgba(120,150,170,.18);padding:16px;border-radius:14px}
.hero{padding:2rem;border-radius:20px;background:linear-gradient(120deg,#10233f,#1b4960,#176b70);color:white}
.hero h1{color:white;font-size:2.7rem;margin:0}.hero p{color:#d5e9ef;margin:.4rem 0 0}
</style>
<div class="hero"><h1>🧠 NeuroSight AI</h1><p>See beyond the scan. Explainable deep learning for brain MRI image classification.</p></div>
""", unsafe_allow_html=True)
st.subheader("Intelligence workspace")
st.write("Explore MRI predictions, model evaluation, visual explanations and dataset quality checks.")
a,b,c,d=st.columns(4)
for col,label,value in zip((a,b,c,d),("Target classes","Model families","Explainability","Benchmark"),("4","3","Grad-CAM","Image-level")): col.metric(label,value)
left,right=st.columns(2)
with left:
    st.markdown("### 🔬 MRI prediction")
    st.write("Upload an MRI image and inspect model scores.")
    st.page_link("pages/1_MRI_Prediction.py",label="Open MRI Prediction →")
    st.markdown("### 📊 Model analytics")
    st.write("Review measured evaluation metrics and confusion matrices.")
    st.page_link("pages/2_Model_Analytics.py",label="Open Model Analytics →")
with right:
    st.markdown("### 🔎 Grad-CAM explainability")
    st.write("Visualize regions that influenced the model output.")
    st.page_link("pages/3_GradCAM_Explainability.py",label="Open Explainability →")
    st.markdown("### 🗂️ Dataset audit")
    st.write("Inspect class balance, exact duplicates and split integrity.")
    st.page_link("pages/4_Dataset_Audit.py",label="Open Dataset Audit →")
st.divider()
st.warning("Research and education only. NeuroSight AI is not a medical device and must not be used for diagnosis or treatment decisions.")
st.caption("A model score is not a calibrated clinical probability. Do not upload identifiable patient information.")
