import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .dataset import CLASS_NAMES
from .evaluate import metrics


MODEL_NAMES = ["cnn", "resnet18", "mobilenet_v2"]


def load_predictions(split):
    records = {}
    for name in MODEL_NAMES:
        path = Path("outputs") / name / f"{split}_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Train all three models with the same data manifest and seed first.")
        data = np.load(path, allow_pickle=False)
        records[name] = {"y_true": data["y_true"], "y_prob": data["y_prob"], "paths": data["paths"]}
    reference = records[MODEL_NAMES[0]]
    for name, record in records.items():
        if not np.array_equal(reference["y_true"], record["y_true"]) or not np.array_equal(reference["paths"], record["paths"]):
            raise ValueError(f"{name} predictions do not align with the CNN split. Retrain all models using the same manifest and seed.")
    return records


def tune_weights(records, step=0.05):
    y_true = records[MODEL_NAMES[0]]["y_true"]
    probabilities = [records[name]["y_prob"] for name in MODEL_NAMES]
    candidates = np.arange(0.0, 1.0001, step)
    best = None
    for w1 in candidates:
        for w2 in candidates:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            weights = np.asarray([w1, w2, max(0.0, w3)])
            combined = sum(w * p for w, p in zip(weights, probabilities))
            score = f1_score(y_true, combined.argmax(axis=1), average="macro", zero_division=0)
            candidate = (float(score), weights)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best[1], best[0]


def main():
    parser = argparse.ArgumentParser(description="Tune a probability-weighted ensemble on validation data and select the overall best system.")
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid resolution; smaller is slower.")
    args = parser.parse_args()
    if not 0 < args.step <= 0.25:
        raise ValueError("--step must be in (0, 0.25].")

    validation = load_predictions("validation")
    weights, validation_f1 = tune_weights(validation, args.step)
    test = load_predictions("test")
    test_probabilities = [test[name]["y_prob"] for name in MODEL_NAMES]
    ensemble_probabilities = sum(w * p for w, p in zip(weights, test_probabilities))
    y_true = test[MODEL_NAMES[0]]["y_true"]
    y_pred = ensemble_probabilities.argmax(axis=1)
    ensemble_result = metrics(y_true, y_pred, CLASS_NAMES, ensemble_probabilities)
    ensemble_result.update({"model": "ensemble", "weights": dict(zip(MODEL_NAMES, weights.tolist())),
                            "validation_macro_f1": validation_f1, "selection_metric": "validation_macro_f1"})
    ensemble_dir = Path("outputs/ensemble")
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    (ensemble_dir / "test_metrics.json").write_text(json.dumps(ensemble_result, indent=2))
    (Path("models/ensemble")).mkdir(parents=True, exist_ok=True)
    config = {"model": "ensemble", "members": MODEL_NAMES, "weights": dict(zip(MODEL_NAMES, weights.tolist())),
              "selection_metric": "validation_macro_f1", "validation_macro_f1": validation_f1}
    (Path("models/ensemble/ensemble_config.json")).write_text(json.dumps(config, indent=2))

    individual_scores = {}
    for name in MODEL_NAMES:
        path = Path("outputs") / name / "training_history.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        history = json.loads(path.read_text())
        individual_scores[name] = float(history["best_validation_macro_f1"])
    candidates = {**individual_scores, "ensemble": float(validation_f1)}
    winner = max(candidates, key=candidates.get)
    overall_dir = Path("models/overall_best")
    overall_dir.mkdir(parents=True, exist_ok=True)
    if winner == "ensemble":
        members = []
        for name in MODEL_NAMES:
            checkpoint = torch.load(Path("models") / name / "best_model.pt", map_location="cpu")
            members.append({"model": name, "state_dict": checkpoint["state_dict"]})
        bundle = {"model": "ensemble", "members": members, "weights": weights.tolist(),
                  "class_names": CLASS_NAMES, "image_size": 224, "selection_metric": "validation_macro_f1"}
        torch.save(bundle, overall_dir / "best_model.pt")
    else:
        shutil.copy2(Path("models") / winner / "best_model.pt", overall_dir / "best_model.pt")

    selection = {
        "overall_best": winner,
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": candidates,
        "ensemble_weights": config["weights"],
        "test_metrics_path": "outputs/ensemble/test_metrics.json" if winner == "ensemble" else f"outputs/{winner}/test_metrics.json",
        "warning": "Test metrics are for final reporting only; model selection and ensemble weights were determined using validation data.",
    }
    (overall_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    (Path("outputs/comparison")).mkdir(parents=True, exist_ok=True)
    rows = [{"model": name, "validation_macro_f1": score, "selected_overall_best": name == winner}
            for name, score in candidates.items()]
    import csv
    with (Path("outputs/comparison/model_comparison.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (Path("outputs/comparison/comparison_report.json")).write_text(json.dumps(selection, indent=2))
    print(json.dumps({"overall_best": winner, "validation_macro_f1": candidates,
                      "ensemble_test_metrics": ensemble_result}, indent=2))


if __name__ == "__main__":
    main()
