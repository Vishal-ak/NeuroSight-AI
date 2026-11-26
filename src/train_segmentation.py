import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.segmentation import UNet,dice_loss,dice_score_from_logits
from src.segmentation_data import MRISegmentationDataset

def evaluate(model,loader,device):
    model.eval(); losses=[]; dices=[]; ious=[]
    with torch.no_grad():
        for image,mask in loader:
            image,mask=image.to(device),mask.to(device)
            logits=model(image)
            losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logits,mask).item()+dice_loss(logits,mask).item()))
            pred=(torch.sigmoid(logits)>=0.5).float()
            dims=(1,2,3); inter=(pred*mask).sum(dims); union=((pred+mask)>0).float().sum(dims)
            dices.extend(((2*inter+1e-7)/(pred.sum(dims)+mask.sum(dims)+1e-7)).cpu().tolist())
            ious.extend(((inter+1e-7)/(union+1e-7)).cpu().tolist())
    return {"loss":float(np.mean(losses)),"dice":float(np.mean(dices)),"iou":float(np.mean(ious))}

def main():
    p=argparse.ArgumentParser(description="Train U-Net on paired MRI images and expert masks.")
    p.add_argument("--manifest",default="data/segmentation/manifest.csv")
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--image-size",type=int,default=256); p.add_argument("--base-channels",type=int,default=32)
    p.add_argument("--workers",type=int,default=2); p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=MRISegmentationDataset(args.manifest,"train",args.image_size)
    val=MRISegmentationDataset(args.manifest,"val",args.image_size)
    test=MRISegmentationDataset(args.manifest,"test",args.image_size)
    opts={"batch_size":args.batch_size,"num_workers":args.workers,"pin_memory":device.type=="cuda"}
    train_loader=DataLoader(train,shuffle=True,**opts)
    val_loader=DataLoader(val,shuffle=False,**opts); test_loader=DataLoader(test,shuffle=False,**opts)
    model=UNet(base_channels=args.base_channels).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    out=Path("outputs/segmentation"); out.mkdir(parents=True,exist_ok=True)
    best=-1.0; history=[]
    for epoch in range(args.epochs):
        model.train(); epoch_losses=[]
        for image,mask in train_loader:
            image,mask=image.to(device),mask.to(device)
            optimizer.zero_grad(set_to_none=True); logits=model(image)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,mask)+dice_loss(logits,mask)
            loss.backward(); optimizer.step(); epoch_losses.append(float(loss.item()))
        val_metrics=evaluate(model,val_loader,device)
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(epoch_losses)),**{f"val_{k}":v for k,v in val_metrics.items()}})
        print(history[-1])
        if val_metrics["dice"]>best:
            best=val_metrics["dice"]
            ck={"state_dict":model.state_dict(),"image_size":args.image_size,"base_channels":args.base_channels,"threshold":0.5,"seed":args.seed}
            torch.save(ck,out/"unet_best.pt"); Path("models/segmentation").mkdir(parents=True,exist_ok=True); torch.save(ck,"models/segmentation/unet_best.pt")
    ck=torch.load(out/"unet_best.pt",map_location=device); model.load_state_dict(ck["state_dict"])
    result={"test":evaluate(model,test_loader,device),"validation_best_dice":best,"history":history,"note":"Segmentation benchmark; inspect source and patient-level split metadata before interpretation."}
    (out/"segmentation_metrics.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["test"],indent=2))

if __name__=="__main__": main()
