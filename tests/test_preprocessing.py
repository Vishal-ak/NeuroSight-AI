from PIL import Image
from src.preprocessing import build_transforms

def test_grayscale_image_converts_to_three_channel_tensor():
    tensor=build_transforms(224)(Image.new("L",(120,150)))
    assert tuple(tensor.shape)==(3,224,224)
