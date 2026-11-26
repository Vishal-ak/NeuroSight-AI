import torch.nn.functional as F

class GradCAM:
    """Minimal Grad-CAM implementation for convolutional classification models."""
    def __init__(self,model,target_layer):
        self.model=model
        self.activations=None
        self.gradients=None
        self.forward_hook=target_layer.register_forward_hook(self._forward)
        self.backward_hook=target_layer.register_full_backward_hook(self._backward)
    def _forward(self,module,inputs,output): self.activations=output
    def _backward(self,module,grad_input,grad_output): self.gradients=grad_output[0]
    def __call__(self,x,class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits=self.model(x)
        if isinstance(logits,tuple): logits=logits[0]
        if class_idx is None: class_idx=int(logits.argmax(1).item())
        logits[:,class_idx].sum().backward()
        weights=self.gradients.mean((2,3),keepdim=True)
        cam=(weights*self.activations).sum(1,keepdim=True).relu()
        cam=F.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False)
        cam=cam-cam.amin((2,3),keepdim=True)
        cam=cam/cam.amax((2,3),keepdim=True).clamp_min(1e-8)
        return cam.detach(),class_idx
    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def target_layer_for(model,model_name):
    if model_name=="cnn": return model.features[14]
    if model_name=="resnet18": return model.layer4[-1].conv2
    if model_name=="mobilenet_v2": return model.features[-1]
    raise ValueError(f"Unsupported Grad-CAM model: {model_name}")
