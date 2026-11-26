import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .dataset import MRIDataset, CLASS_NAMES
from .preprocessing import build_transforms
from .models import build_model
from .evaluate import metrics


def reject_output_images(path):
    resolved = Path(path).expanduser().resolve()
    if any(part.lower() == "output_images" for part in resolved.parts):
        raise ValueError("Output_images is a previous-run output folder, not a training dataset. Use the original Training/Testing dataset.")
    return resolved


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    y_true, y_pred, y_prob = [], [], []
    total_loss, count = 0.0, 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            count += len(labels)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
            y_prob.extend(probabilities.detach().cpu().numpy().tolist())
    elapsed = time.perf_counter() - started
    return {
        "loss": total_loss / max(count, 1),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / max(count, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Train one NeuroSight classifier and save reproducible experiment artifacts.")
    parser.add_argument("--data-dir", default="data/raw/Brain_Tumor_MRI_Dataset")
    parser.add_argument("--model", choices=["cnn", "resnet18", "mobilenet_v2"], default="cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    data_dir = reject_output_images(args.data_dir)
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run scripts/audit_dataset.py against the ORIGINAL dataset first.")
    import pandas as pd
    manifest_df = pd.read_csv(manifest)
    if "path" not in manifest_df.columns or manifest_df.empty:
        raise ValueError("The dataset manifest is empty or missing its path column.")
    if manifest_df.path.astype(str).str.contains(r"(?i)(^|[/\\])output_images([/\\]|$)", regex=True).any():
        raise ValueError("Manifest contains Output_images entries. Rebuild it from the original Training/Testing folders.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plain = MRIDataset(manifest, "train", build_transforms(args.image_size))
    labels = plain.df.label_id.to_numpy()
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)), test_size=0.15, random_state=args.seed, stratify=labels
    )
    augmented = MRIDataset(manifest, "train", build_transforms(args.image_size, True))
    train_set = Subset(augmented, train_indices)
    val_set = Subset(plain, val_indices)
    test_set = MRIDataset(manifest, "test", build_transforms(args.image_size))
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model = build_model(args.model, len(CLASS_NAMES), not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    out = Path("outputs") / args.model
    model_dir = Path("models") / args.model
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0
    history = []

    for epoch in range(args.epochs):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device)
        val_metrics = metrics(val_stats["y_true"], val_stats["y_pred"], CLASS_NAMES, val_stats["y_prob"])
        record = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "validation_loss": val_stats["loss"],
            "validation_accuracy": val_stats["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss {record['train_loss']:.4f} | "
            f"train acc {record['train_accuracy']:.4f} | val loss {record['validation_loss']:.4f} | "
            f"val acc {record['validation_accuracy']:.4f} | val macro-F1 {record['validation_macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch + 1
            checkpoint = {
                "state_dict": model.state_dict(), "model": args.model,
                "class_names": CLASS_NAMES, "image_size": args.image_size,
                "seed": args.seed, "best_epoch": best_epoch,
            }
            torch.save(checkpoint, out / "best_model.pt")
            torch.save(checkpoint, model_dir / "best_model.pt")
            np.savez_compressed(
                out / "validation_predictions.npz",
                y_true=np.asarray(val_stats["y_true"], dtype=np.int64),
                y_prob=np.asarray(val_stats["y_prob"], dtype=np.float32),
                paths=np.asarray(plain.df.iloc[val_indices].path.astype(str).tolist()),
            )

    (out / "training_history.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "epochs_requested": args.epochs,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "hyperparameters": {"batch_size": args.batch_size, "image_size": args.image_size,
                            "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
                            "pretrained": not args.no_pretrained},
        "history": history,
    }, indent=2))

    checkpoint = torch.load(out / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_stats = run_epoch(model, test_loader, criterion, device)
    inference_ms = 1000.0 * test_stats["elapsed_seconds"] / max(len(test_stats["y_true"]), 1)
    result = metrics(
        test_stats["y_true"], test_stats["y_pred"], CLASS_NAMES, test_stats["y_prob"],
        inference_ms_per_image=inference_ms, checkpoint_path=out / "best_model.pt"
    )
    result.update({"model": args.model, "best_epoch": best_epoch, "selection_metric": "validation_macro_f1"})
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        out / "test_predictions.npz",
        y_true=np.asarray(test_stats["y_true"], dtype=np.int64),
        y_prob=np.asarray(test_stats["y_prob"], dtype=np.float32),
        paths=np.asarray(test_set.df.path.astype(str).tolist()),
    )
    print("Held-out deduplicated image-level test metrics:")
    print(json.dumps({k: v for k, v in result.items() if k != "classification_report"}, indent=2))
    print("No patient identifiers are available; this is not patient-level or clinical validation.")


if __name__ == "__main__":
    main()
