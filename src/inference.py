import torch
from .models import build_model
from .preprocessing import build_transforms


def _load_model(model_name, state_dict, class_count, device):
    model = build_model(model_name, class_count, False)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_predictor(checkpoint, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device)
    class_names = ck["class_names"]
    transform = build_transforms(ck.get("image_size", 224))

    if ck.get("model") == "ensemble":
        members = [
            (_load_model(member["model"], member["state_dict"], len(class_names), device), float(weight))
            for member, weight in zip(ck["members"], ck["weights"])
        ]

        def predict(pil_image):
            x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = sum(weight * torch.softmax(model(x), dim=1) for model, weight in members)[0]
                probabilities = probabilities.cpu().tolist()
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            return {"predicted_class": class_names[index], "probabilities": dict(zip(class_names, probabilities)),
                    "model_used": "Optimized ensemble"}
        return predict

    model = _load_model(ck["model"], ck["state_dict"], len(class_names), device)

    def predict(pil_image):
        x = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(x), 1)[0].cpu().tolist()
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"predicted_class": class_names[index], "probabilities": dict(zip(class_names, probabilities)),
                "model_used": ck["model"]}
    return predict
