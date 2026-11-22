"""Measure a frozen teacher on the original training subset without augmentation."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Subset


def distribution(values: np.ndarray) -> dict:
    """Summarize the empirical distribution without approximating quantiles."""
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty array of finite probabilities")
    counts, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, 21))
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
        "histogram_counts": counts.tolist(),
        "histogram_edges": edges.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--omitted-digit", type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(args.project.resolve()))
    from data import make_datasets, make_loader
    from train import file_hash, load_teacher, seed_everything, write_json

    saved = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = dict(saved.get("config", saved))
    cfg.update(download=False, workers=0, device="cuda")
    if args.data_root is not None:
        cfg["data_root"] = str(args.data_root.resolve())
    if args.teacher is not None:
        cfg["teacher"] = str(args.teacher.resolve())
    temperature = cfg["temperature"] if args.temperature is None else args.temperature
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if not cfg.get("teacher"):
        raise ValueError("Use a distillation config, or supply --teacher")
    classes = 100 if cfg["dataset"] == "cifar100" else 10
    omitted = cfg.get("omit") if args.omitted_digit is None else args.omitted_digit
    if omitted is not None and not 0 <= omitted < classes:
        raise ValueError("Omitted class is outside the model's output classes")

    seed_everything(cfg["seed"], cfg["deterministic"])
    device = torch.device(cfg["device"])
    teacher, _ = load_teacher(cfg, classes, device)
    (train, validation, _), data_info = make_datasets(cfg)
    result_path = args.config.parent / "result.json"
    if result_path.exists():
        original = json.loads(result_path.read_text(encoding="utf-8"))
        if original["data"] != data_info:
            raise ValueError("Training subset differs from the original run")
        if original.get("teacher") and original["teacher"]["sha256"] != file_hash(cfg["teacher"]):
            raise ValueError("Teacher differs from the original run")
    # The validation dataset uses the same source images with plain preprocessing.
    # Select the TRAIN indices from it to disable augmentation without changing data.
    plain_train = Subset(validation.dataset, list(train.indices))
    loader = make_loader(plain_train, cfg)
    labels, weights, omitted_probabilities = [], [], []
    teacher_correct = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = teacher(images)
            softened = (logits / temperature).softmax(dim=1)
            target_probability = softened.gather(1, targets[:, None]).squeeze(1)
            labels.append(targets.cpu().numpy())
            weights.append((1.0 - target_probability).cpu().numpy())
            teacher_correct += int((logits.argmax(1) == targets).sum().item())
            if omitted is not None:
                non_target_logits = (logits / temperature).scatter(
                    1, targets[:, None], float("-inf")
                )
                conditional = non_target_logits.softmax(dim=1)[:, omitted]
                omitted_probabilities.append(conditional.cpu().numpy())

    targets = np.concatenate(labels)
    values = np.concatenate(weights).astype(np.float64)
    beta = float(cfg["beta"])
    summary = {
        "dataset": cfg["dataset"],
        "profile": cfg["profile"],
        "split_seed": cfg["split_seed"],
        "temperature": temperature,
        "preprocessing": "original training indices, normalized, no crop or flip",
        "data": data_info,
        "teacher_sha256": file_hash(cfg["teacher"]),
        "teacher_training_subset_accuracy": teacher_correct / len(plain_train),
        "one_minus_p_y": distribution(values),
        "dkd_beta": beta,
        "beta_over_mean_one_minus_p_y": beta / values.mean(),
        "coefficient_note": (
            "Coefficients inside the decomposition, before global distillation "
            "weight and warmup. Coefficient ratios are not gradient ratios."
        ),
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    conditional_values = None
    if omitted is not None:
        conditional_values = np.concatenate(omitted_probabilities)
        valid = targets != omitted
        summary["omitted_class"] = omitted
        summary["omitted_class_conditional_non_target_probability"] = distribution(
            conditional_values[valid].astype(np.float64)
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "teacher_confidence.json", summary)
    with (args.output / "teacher_confidence_samples.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        columns = ["source_train_index", "label", "one_minus_p_y"]
        if conditional_values is not None:
            columns.append("omitted_class_conditional_non_target_probability")
        writer.writerow(columns)
        for position, source_index in enumerate(plain_train.indices):
            row = [int(source_index), int(targets[position]), float(values[position])]
            if conditional_values is not None:
                value = "" if targets[position] == omitted else float(conditional_values[position])
                row.append(value)
            writer.writerow(row)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
