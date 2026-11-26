from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Analytics | NeuroSight AI",page_icon="📊",layout="wide")
st.title("📊 Model Analytics")
st.caption("Metrics are loaded from completed training runs. No placeholder scores are shown.")
selection_path=Path("models/overall_best/selection.json")
if selection_path.exists():
    selection=json.loads(selection_path.read_text())
    st.success(f"Selected overall best: {selection['overall_best']} (validation macro-F1 selection)")
else:
    st.info("Overall-best selection is not available yet. Train all three models and run python -m src.ensemble.")
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
selected=st.selectbox("Inspect evaluated model",df.Model.tolist())
m=df.loc[df.Model==selected,"data"].iloc[0]
a,b,c=st.columns(3)
a.metric("Accuracy",f"{m['accuracy']:.2%}")
b.metric("Macro F1",f"{m['macro_f1']:.3f}")
c.metric("Weighted F1",f"{m['weighted_f1']:.3f}")
st.subheader("Model comparison")
long=df.melt(id_vars="Model",value_vars=["Accuracy","Macro F1","Weighted F1"],var_name="Metric",value_name="Score")
st.plotly_chart(px.bar(long,x="Model",y="Score",color="Metric",barmode="group",range_y=[0,1]),use_container_width=True)
if m.get("inference_ms_per_image") is not None:
    st.caption(f"Measured evaluation throughput: {m['inference_ms_per_image']:.3f} ms/image (hardware-dependent)")
if m.get("estimated_inference_ms_per_image") is not None:
    st.caption(f"Estimated ensemble inference: {m['estimated_inference_ms_per_image']:.3f} ms/image; benchmark on deployment hardware before comparing latency.")
if m.get("checkpoint_size_mb") is not None:
    st.caption(f"Checkpoint size: {m['checkpoint_size_mb']:.2f} MB")
report=pd.DataFrame(m.get("classification_report",{})).T
if not report.empty:
    report=report.drop(index=[x for x in ["accuracy","macro avg","weighted avg"] if x in report.index],errors="ignore")
    st.subheader("Class-wise precision, recall and F1")
    st.dataframe(report.style.format("{:.3f}"),use_container_width=True)
per_class=m.get("per_class",{})
if per_class:
    st.subheader("Sensitivity and specificity by class")
    st.dataframe(pd.DataFrame(per_class).T.style.format("{:.3f}"),use_container_width=True)
if "confusion_matrix" in m:
    names=["glioma","meningioma","notumor","pituitary"]
    cm=pd.DataFrame(m["confusion_matrix"],index=names,columns=names)
    st.subheader("Confusion matrix")
    st.plotly_chart(px.imshow(cm,text_auto=True,x=names,y=names,labels={"x":"Predicted","y":"Actual","color":"Count"},color_continuous_scale="Blues"),use_container_width=True)
history_path=Path("outputs")/selected/"training_history.json"
if history_path.exists():
    history=json.loads(history_path.read_text()).get("history",[])
    if history:
        history_df=pd.DataFrame(history)
        st.subheader("Training curves")
        curve=history_df.melt(id_vars="epoch",value_vars=["train_loss","validation_loss"],var_name="Series",value_name="Loss")
        st.plotly_chart(px.line(curve,x="epoch",y="Loss",color="Series",markers=True),use_container_width=True)
        acc=history_df.melt(id_vars="epoch",value_vars=["train_accuracy","validation_accuracy"],var_name="Series",value_name="Accuracy")
        st.plotly_chart(px.line(acc,x="epoch",y="Accuracy",color="Series",markers=True),use_container_width=True)
        st.line_chart(history_df.set_index("epoch")["validation_macro_f1"])
st.warning("The supplied collection contains exact duplicates across its original train/test folders. Results are image-level, not patient-level or clinical validation.")
