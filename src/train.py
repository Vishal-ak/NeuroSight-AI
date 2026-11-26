import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from .dataset import MRIDataset,CLASS_NAMES
from .preprocessing import build_transforms
from .models import build_model
from .evaluate import metrics

def run_epoch(model,loader,loss_fn,device,optimizer=None):
    training=optimizer is not None
    model.train(training)
    y_true=[]; y_pred=[]; total_loss=0.0; count=0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for images,labels in loader:
            images,labels=images.to(device),labels.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=loss_fn(logits,labels)
            if training:
                loss.backward()
                optimizer.step()
            predictions=logits.argmax(1)
            total_loss+=loss.item()*len(labels)
            count+=len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    return total_loss/max(count,1),sum(a==b for a,b in zip(y_true,y_pred))/max(count,1),y_true,y_pred

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data/raw/Brain_Tumor_MRI_Dataset")
    parser.add_argument("--model",choices=["cnn","resnet18","mobilenet_v2"],default="cnn")
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--image-size",type=int,default=224)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--no-pretrained",action="store_true")
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest=Path(args.data_dir)/"manifest.csv"
    if not manifest.exists(): raise FileNotFoundError("Run scripts/audit_dataset.py first.")
    plain=MRIDataset(manifest,"train",build_transforms(args.image_size))
    labels=plain.df.label_id.to_numpy()
    train_indices,val_indices=train_test_split(np.arange(len(labels)),test_size=0.15,random_state=args.seed,stratify=labels)
    augmented=MRIDataset(manifest,"train",build_transforms(args.image_size,True))
    train_set=Subset(augmented,train_indices)
    val_set=Subset(plain,val_indices)
    test_set=MRIDataset(manifest,"test",build_transforms(args.image_size))
    loader_args={"batch_size":args.batch_size,"num_workers":args.workers,"pin_memory":device.type=="cuda"}
    train_loader=DataLoader(train_set,shuffle=True,**loader_args)
    val_loader=DataLoader(val_set,shuffle=False,**loader_args)
    test_loader=DataLoader(test_set,shuffle=False,**loader_args)
    model=build_model(args.model,len(CLASS_NAMES),not args.no_pretrained).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    out=Path("outputs")/args.model
    out.mkdir(parents=True,exist_ok=True)
    best_f1=-1.0
    for epoch in range(args.epochs):
        train_stats=run_epoch(model,train_loader,criterion,device,optimizer)
        val_stats=run_epoch(model,val_loader,criterion,device)
        val_metrics=metrics(val_stats[2],val_stats[3],CLASS_NAMES)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_stats[0]:.4f} | train acc {train_stats[1]:.4f} | val loss {val_stats[0]:.4f} | val acc {val_stats[1]:.4f} | val macro-F1 {val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"]>best_f1:
            best_f1=val_metrics["macro_f1"]
            checkpoint={"state_dict":model.state_dict(),"model":args.model,"class_names":CLASS_NAMES,"image_size":args.image_size,"seed":args.seed}
            torch.save(checkpoint,out/"best_model.pt")
            app_checkpoint=Path("models")/args.model/"best_model.pt"
            app_checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save(checkpoint,app_checkpoint)
    checkpoint=torch.load(out/"best_model.pt",map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_stats=run_epoch(model,test_loader,criterion,device)
    result=metrics(test_stats[2],test_stats[3],CLASS_NAMES)
    (out/"test_metrics.json").write_text(json.dumps(result,indent=2))
    print("Deduplicated image-level test metrics:")
    print(json.dumps({k:v for k,v in result.items() if k!="classification_report"},indent=2))
    print("No patient identifiers are available in this dataset; do not describe this as patient-level or clinical validation.")

if __name__=="__main__": main()
