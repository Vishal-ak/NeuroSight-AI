from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from src.models import build_model
from src.preprocessing import build_transforms
from src.explainability import GradCAM,target_layer_for

st.set_page_config(page_title="Grad-CAM | NeuroSight AI",page_icon="🔎",layout="wide")
st.title("🔎 Grad-CAM Explainability")
st.caption("Grad-CAM highlights image regions associated with a model output. It is a qualitative debugging aid, not clinical evidence.")
paths=sorted(list(Path("models").glob("**/*.pt"))+list(Path("models").glob("**/*.pth")))
if not paths:
    st.info("No trained checkpoint found. Train a model and put its checkpoint under models/.")
    st.stop()
checkpoint=st.selectbox("Checkpoint",paths,format_func=str)
upload=st.file_uploader("MRI image",type=["jpg","jpeg","png","bmp"])
if upload:
    image=Image.open(upload).convert("RGB")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ck=torch.load(checkpoint,map_location=device)
        model=build_model(ck["model"],len(ck["class_names"]),False).to(device)
        model.load_state_dict(ck["state_dict"]); model.eval()
        x=build_transforms(ck.get("image_size",224))(image).unsqueeze(0).to(device)
        engine=GradCAM(model,target_layer_for(model,ck["model"]))
        cam,index=engine(x); engine.close()
        with torch.no_grad(): scores=torch.softmax(model(x),1)[0].cpu().tolist()
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].imshow(image); ax[0].set_title("Input MRI"); ax[0].axis("off")
        ax[1].imshow(image); ax[1].imshow(cam[0,0].cpu().numpy(),cmap="jet",alpha=.42); ax[1].set_title("Grad-CAM overlay"); ax[1].axis("off")
        st.pyplot(fig,clear_figure=True)
        st.metric("Predicted class",ck["class_names"][index].title())
        st.caption(" · ".join(f"{name}: {score:.1%}" for name,score in zip(ck["class_names"],scores)))
    except Exception as exc: st.error(f"Could not generate Grad-CAM: {exc}")
st.warning("Heatmaps can be unstable or misleading. They do not reliably segment tumors or establish a diagnosis.")
