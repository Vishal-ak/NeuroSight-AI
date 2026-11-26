from pathlib import Path
import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Audit | NeuroSight AI",page_icon="🗂️",layout="wide")
st.title("🗂️ Dataset Audit")
st.caption("Quality and split-integrity reports produced by scripts/audit_dataset.py.")
summary=Path("outputs/audit/audit_summary.json")
manifest=Path("data/raw/Brain_Tumor_MRI_Dataset/manifest.csv")
if not summary.exists():
    st.info("No audit report found. Extract the dataset and run the audit script.")
    st.code("python scripts/audit_dataset.py --data-dir data/raw/Brain_Tumor_MRI_Dataset")
    st.stop()
s=json.loads(summary.read_text())
a,b,c,d=st.columns(4)
a.metric("Readable images",s.get("total_valid_images",0))
b.metric("Unique hashes",s.get("unique_hashes",0))
c.metric("Duplicate groups",s.get("duplicate_groups",0))
d.metric("Cross-split groups",s.get("cross_split_duplicate_groups",0))
st.subheader("Audit details"); st.json(s)
if manifest.exists():
    df=pd.read_csv(manifest)
    usable=df[df["usable"].astype(bool)]
    counts=usable.groupby(["partition","class_name"]).size().unstack(fill_value=0)
    st.subheader("Usable class distribution"); st.bar_chart(counts)
    st.subheader("Manifest preview"); st.dataframe(df.head(100),use_container_width=True)
    st.download_button("Download manifest CSV",df.to_csv(index=False),file_name="manifest.csv",mime="text/csv")
else: st.warning("Manifest not found at the configured data path.")
