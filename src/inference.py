import torch
from .models import build_model
from .preprocessing import build_transforms

def load_predictor(checkpoint,device=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(checkpoint,map_location=device)
    model=build_model(ck["model"],len(ck["class_names"]),False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    transform=build_transforms(ck.get("image_size",224))
    def predict(pil_image):
        x=transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities=torch.softmax(model(x),1)[0].cpu().tolist()
        index=max(range(len(probabilities)),key=probabilities.__getitem__)
        return {"predicted_class":ck["class_names"][index],"probabilities":dict(zip(ck["class_names"],probabilities))}
    return predict
