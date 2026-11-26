from torchvision import transforms

MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)

def build_transforms(size=224,train=False):
    ops=[transforms.Resize((size,size))]
    if train:
        ops += [transforms.RandomRotation(7),transforms.RandomHorizontalFlip(0.5)]
    ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]
    return transforms.Compose(ops)
