from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Analytics | NeuroSight AI",page_icon="📊",layout="wide")
st.title("📊 Model Analytics")
st.caption("This page reads actual test_metrics.json files. It does not display placeholder performance.")
paths=sorted(Path("outputs").glob("*/test_metrics.json"))
if not paths:
    st.info("No evaluation results found. Train and evaluate a model first.")
    st.code("python -m src.train --data-dir data/raw/Brain_Tumor_MRI_Dataset --model resnet18 --epochs 15")
    st.stop()
rows=[]
for p in paths:
    try:
        m=json.loads(p.read_text())
        rows.append({"Model":p.parent.name,"Accuracy":m["accuracy"],"Macro F1":m["macro_f1"],"Weighted F1":m["weighted_f1"],"data":m})
    except Exception as exc: st.warning(f"Skipped {p}: {exc}")
if not rows: st.error("No valid metrics files."); st.stop()
df=pd.DataFrame(rows)
selected=st.selectbox("Inspect model",df.Model.tolist())
m=df.loc[df.Model==selected,"data"].iloc[0]
a,b,c=st.columns(3)
a.metric("Accuracy",f"{m['accuracy']:.2%}")
b.metric("Macro F1",f"{m['macro_f1']:.3f}")
c.metric("Weighted F1",f"{m['weighted_f1']:.3f}")
st.subheader("Model comparison")
long=df.melt(id_vars="Model",value_vars=["Accuracy","Macro F1","Weighted F1"],var_name="Metric",value_name="Score")
st.plotly_chart(px.bar(long,x="Model",y="Score",color="Metric",barmode="group",range_y=[0,1]),use_container_width=True)
report=pd.DataFrame(m.get("classification_report",{})).T
if not report.empty:
    report=report.drop(index=[x for x in ["accuracy","macro avg","weighted avg"] if x in report.index],errors="ignore")
    st.subheader("Class-wise report")
    st.dataframe(report.style.format("{:.3f}"),use_container_width=True)
if "confusion_matrix" in m:
    names=["glioma","meningioma","notumor","pituitary"]
    cm=pd.DataFrame(m["confusion_matrix"],index=names,columns=names)
    st.subheader("Confusion matrix")
    st.plotly_chart(px.imshow(cm,text_auto=True,x=names,y=names,labels={"x":"Predicted","y":"Actual","color":"Count"},color_continuous_scale="Blues"),use_container_width=True)
st.warning("The supplied collection has exact duplicates across its original train/test folders. This benchmark is image-level, not patient-level or clinical validation.")
