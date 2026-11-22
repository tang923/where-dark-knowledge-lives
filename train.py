"""一个训练入口：配置 -> 数据 -> 学生/冻结教师 -> 训练 -> 验证选择 -> 测试。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from data import make_datasets, make_loader
from losses import objective
from models import complexity, make_model


def default_config(dataset="mnist"):
    mnist = dataset == "mnist"
    return dict(dataset=dataset, model="mlp800" if mnist else "resnet20",
                method="ce", teacher=None, teacher_model="mlp1200" if mnist else "resnet56",
                teacher_source=None, teacher_training_data="unknown", seed=0, split_seed=661,
                omit=None, selection="best", epochs=20 if mnist else 240,
                batch_size=128 if mnist else 64, lr=0.01 if mnist else 0.05,
                momentum=0.9, weight_decay=0.0 if mnist else 0.0005,
                milestones=[10, 15] if mnist else [150, 180, 210], gamma=0.1,
                temperature=4.0, alpha=1.0, beta=2.0, ce_weight=1.0,
                distill_weight=1.0, warmup=0 if mnist else 20,
                data_root="data", out="runs/manual", device="auto", workers=0,
                download=True, train_limit=0, eval_limit=0, profile="full",
                role="student", comparison="manual", variant="ce", deterministic=True,
                mnist_shift=0, test_validation_floor=None)


def seed_everything(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, obj):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))


def metrics_from_confusion(cm, omitted=None):
    cm = cm.cpu()
    total = int(cm.sum())
    correct = int(cm.diag().sum())
    result = {"accuracy": correct / total, "count": total, "confusion_matrix": cm.tolist()}
    if omitted is not None:
        count = int(cm[omitted].sum())
        hits = int(cm[omitted, omitted])
        result.update(omitted_recall=hits / count if count else None,
                      other_accuracy=(correct - hits) / (total - count) if total > count else None)
    return result


@torch.no_grad()
def evaluate(model, loader, device, classes, omitted=None, *, track_omitted=False):
    model.eval()
    cm = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    track_omitted = track_omitted and omitted is not None
    if track_omitted:
        sums = torch.zeros(3, dtype=torch.float64, device=device)
        other_classes = [c for c in range(classes) if c != omitted]
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(1)
        cm += torch.bincount(targets * classes + predictions,
                             minlength=classes * classes).reshape(classes, classes)
        if track_omitted:
            missing = logits[targets == omitted]
            target_logits = missing[:, omitted]
            sums += torch.stack((missing.softmax(1)[:, omitted], target_logits,
                                 target_logits - missing[:, other_classes].amax(1))).sum(1, dtype=torch.float64)
    metrics = metrics_from_confusion(cm, omitted)
    if track_omitted:
        count = sum(metrics["confusion_matrix"][omitted])
        metrics.update(zip(("omitted_mean_probability", "omitted_mean_logit", "omitted_mean_margin"),
                           (sums / count).tolist() if count else [None] * 3))
    return metrics


def validate_config(cfg):
    if cfg["dataset"] not in ("mnist", "cifar10", "cifar100"):
        raise ValueError("Unknown dataset")
    if cfg["method"] not in ("ce", "kd", "tckd", "nckd", "dkd"):
        raise ValueError("Unknown loss method")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["temperature"] <= 0 or cfg["lr"] <= 0:
        raise ValueError("epochs, batch_size, temperature and lr must be positive")
    if min(cfg[k] for k in ("ce_weight", "distill_weight", "alpha", "beta", "warmup", "workers", "train_limit", "eval_limit")) < 0:
        raise ValueError("Weights and counts must be non-negative")
    if cfg["method"] == "ce" and cfg["ce_weight"] == 0:
        raise ValueError("CE baseline needs nonzero CE weight")
    if cfg["method"] != "ce" and not cfg["teacher"]:
        raise ValueError("Set --teacher for distillation")
    if cfg["method"] == "ce" and cfg["teacher"]:
        raise ValueError("CE baseline should not load a teacher")
    if cfg["ce_weight"] == 0 and (cfg["method"] not in ("kd", "dkd") or
                                    (cfg["method"] == "dkd" and cfg["alpha"] <= 0)):
        raise ValueError("Proposal CE-free controls use KD or DKD with alpha > 0")
    if cfg["omit"] is not None:
        if cfg["dataset"] != "mnist" or not 0 <= cfg["omit"] <= 9:
            raise ValueError("The planned omitted-class study uses one MNIST digit")
        if cfg["selection"] != "last":
            raise ValueError("Use the fixed final epoch for omitted-class runs")
    if cfg["selection"] not in ("best", "last"):
        raise ValueError("selection must be best or last")
    if cfg["profile"] == "full" and (cfg["train_limit"] or cfg["eval_limit"]):
        raise ValueError("Limited data must be labelled pilot/smoke, not full")
    shift = cfg.get("mnist_shift", 0)
    if shift not in (0, 2) or (shift and cfg["dataset"] != "mnist"):
        raise ValueError("mnist_shift must be 0 or 2 and applies only to MNIST")
    floor = cfg.get("test_validation_floor")
    if floor is not None and not 0 <= floor <= 1:
        raise ValueError("test_validation_floor must be between 0 and 1")


def load_teacher(cfg, classes, device):
    checkpoint = torch.load(cfg["teacher"], map_location="cpu", weights_only=True)
    metadata = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if metadata:
        if metadata["dataset"] != cfg["dataset"] or metadata["model"] != cfg["teacher_model"]:
            raise ValueError("Teacher dataset/architecture does not match")
        if metadata.get("omit") is not None:
            raise ValueError("Teacher must be trained on all classes")
        if metadata.get("split_seed") != cfg["split_seed"]:
            raise ValueError("Local teacher must use the same held-out validation split")
        if metadata.get("profile") != cfg["profile"]:
            raise ValueError("Do not mix smoke/pilot teachers with full experiments")
    elif not cfg["teacher_source"]:
        raise ValueError("External checkpoint needs --teacher-source attribution")
    state = checkpoint
    for key in ("model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    teacher = make_model(cfg["teacher_model"], classes).to(device)
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False).eval()
    return teacher, metadata


def write_history(path, history):
    columns = list(dict.fromkeys(key for row in history for key in row))
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    temp.replace(path)


def run(config, *, stop_after=None):
    """自动续训/复用相同配置；不同配置不能覆盖已有实验。stop_after 用于续训测试。"""
    cfg = default_config(config.get("dataset", "mnist")) | config
    unknown = set(cfg) - set(default_config())
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    validate_config(cfg)
    # New optional settings must not change the configurations or fingerprints of old runs.
    if not cfg["mnist_shift"]:
        cfg.pop("mnist_shift")
    if cfg["test_validation_floor"] is None:
        cfg.pop("test_validation_floor")
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["teacher"]:
        cfg["teacher"] = str(Path(cfg["teacher"]).resolve())
    teacher_sha = file_hash(cfg["teacher"]) if cfg["teacher"] else None
    fingerprint_cfg = {k: v for k, v in cfg.items() if k not in ("out", "download", "workers")}
    fingerprint = hashlib.sha256(json.dumps([fingerprint_cfg, teacher_sha], sort_keys=True).encode()).hexdigest()
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError(f"Different experiment already exists in {out}; use a new output directory")
        if (out / "result.json").exists():
            print(f"[cached] {out}", flush=True)
            return json.loads((out / "result.json").read_text(encoding="utf-8"))
    write_json(out / "config.json", {"fingerprint": fingerprint, "config": cfg})
    seed_everything(cfg["seed"], cfg["deterministic"])
    classes = 100 if cfg["dataset"] == "cifar100" else 10
    shape = (1, 28, 28) if cfg["dataset"] == "mnist" else (3, 32, 32)
    device = torch.device(cfg["device"])
    datasets, data_info = make_datasets(cfg)
    train_set, val_set, test_set = datasets
    val_loader = make_loader(val_set, cfg)
    model = make_model(cfg["model"], classes).to(device)
    initial_hash = model_hash(model)
    model_complexity = complexity(model, shape)
    teacher, teacher_info = None, None
    if cfg["teacher"]:
        teacher, teacher_cfg = load_teacher(cfg, classes, device)
        teacher_info = dict(sha256=teacher_sha, source=cfg["teacher_source"] or "local training",
                            training_data=cfg["teacher_training_data"], config=teacher_cfg,
                            validation=evaluate(teacher, val_loader, device, classes),
                            complexity=complexity(teacher, shape))
        print(f"Teacher validation: {teacher_info['validation']['accuracy']:.4f}", flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"])
    start, best_score, selected_epoch, elapsed, history = 0, -1.0, 0, 0.0, []
    if (out / "last.pt").exists():
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["fingerprint"] != fingerprint or checkpoint["data_info"] != data_info:
            raise ValueError("Resume configuration/data mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        start, best_score = checkpoint["epoch"], checkpoint["best_score"]
        selected_epoch, elapsed, history = checkpoint["selected_epoch"], checkpoint["elapsed"], checkpoint["history"]
        if checkpoint["initial_hash"] != initial_hash:
            raise ValueError("Paired initialization changed")
        print(f"[resume] {out}: epoch {start}", flush=True)
    for epoch in range(start, cfg["epochs"]):
        tick = time.perf_counter()
        model.train()
        # 每轮重新设种子，使增强、dropout 和断点续训可复现。
        seed_everything(cfg["seed"] * 100000 + epoch + 1, cfg["deterministic"])
        sums = torch.zeros(4, device=device)
        lr = optimizer.param_groups[0]["lr"]
        warmup = min((epoch + 1) / cfg["warmup"], 1.0) if cfg["warmup"] else 1.0
        for images, targets in make_loader(train_set, cfg, train=True, epoch=epoch + 1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images) if teacher is not None else None
            logits = model(images)
            loss, ce, distill = objective(logits, targets, teacher_logits, method=cfg["method"],
                                        temperature=cfg["temperature"], ce_weight=cfg["ce_weight"],
                                        distill_weight=cfg["distill_weight"] * warmup,
                                        alpha=cfg["alpha"], beta=cfg["beta"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            sums += torch.stack((loss.detach() * len(targets), ce * len(targets),
                                 distill * len(targets), (logits.argmax(1) == targets).sum()))
        validation = evaluate(model, val_loader, device, classes, cfg["omit"], track_omitted=True)
        values = (sums / len(train_set)).cpu().tolist()
        seconds = time.perf_counter() - tick
        elapsed += seconds
        row = dict(epoch=epoch + 1, lr=lr, loss=values[0], ce=values[1], distill=values[2],
                   train_accuracy=values[3], val_accuracy=validation["accuracy"], seconds=seconds)
        if cfg["omit"] is not None:
            for key in ("omitted_recall", "omitted_mean_probability", "omitted_mean_logit", "omitted_mean_margin"):
                row[key] = validation[key]
        history.append(row)
        scheduler.step()
        if validation["accuracy"] > best_score:
            best_score = validation["accuracy"]
            if cfg["selection"] == "best":
                selected_epoch = epoch + 1
                save_checkpoint(out / "best.pt", {"model": model.state_dict(), "config": cfg})
        if cfg["selection"] == "last":
            selected_epoch = epoch + 1
            save_checkpoint(out / "best.pt", {"model": model.state_dict(), "config": cfg})
        save_checkpoint(out / "last.pt", dict(model=model.state_dict(), config=cfg,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                        epoch=epoch + 1, best_score=best_score, selected_epoch=selected_epoch,
                        elapsed=elapsed, history=history, fingerprint=fingerprint,
                        data_info=data_info, initial_hash=initial_hash))
        write_history(out / "history.csv", history)
        print(f"{out.name} {epoch+1}/{cfg['epochs']} loss={values[0]:.4f} "
              f"val={validation['accuracy']:.4f} {seconds:.1f}s", flush=True)
        if stop_after and epoch + 1 >= stop_after and epoch + 1 < cfg["epochs"]:
            return None
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    validation = evaluate(model, val_loader, device, classes, cfg["omit"])
    test_validation_floor = cfg.get("test_validation_floor")
    test = None
    if test_validation_floor is None or validation["accuracy"] >= test_validation_floor:
        test = evaluate(model, make_loader(test_set, cfg), device, classes, cfg["omit"])
    write_history(out / "history.csv", history)
    result = dict(config=cfg, fingerprint=fingerprint, initial_hash=initial_hash, data=data_info,
                  complexity=model_complexity, teacher=teacher_info, selected_epoch=selected_epoch,
                  validation=validation, test=test,
                  training_seconds=elapsed, history=history,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                   device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
                  exploratory=cfg["profile"] != "full")
    write_json(out / "result.json", result)
    score = f"test={test['accuracy']:.4f}" if test else f"val={validation['accuracy']:.4f}; test skipped"
    print(f"[done] {out}: {score}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="mnist")
    parser.add_argument("--config", help="Optional JSON config (same keys as default_config)")
    # 保持单次运行入口简单；完整矩阵由 experiments.py 生成。
    for name in ("model", "method", "teacher", "teacher-model", "teacher-source", "teacher-training-data",
                 "out", "data-root", "device", "profile", "selection"):
        parser.add_argument("--" + name)
    for name in ("seed", "epochs", "batch-size", "workers", "omit", "warmup", "train-limit", "eval-limit", "mnist-shift"):
        parser.add_argument("--" + name, type=int)
    for name in ("lr", "temperature", "ce-weight", "distill-weight", "alpha", "beta", "test-validation-floor"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--no-download", action="store_true")
    args = vars(parser.parse_args())
    path = args.pop("config")
    no_download = args.pop("no_download")
    loaded = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    # --config 可直接使用运行目录里的 config.json。
    loaded = loaded.get("config", loaded)
    dataset = loaded.get("dataset", args["dataset"])
    cfg = default_config(dataset) | loaded
    cfg.update({k: v for k, v in args.items() if v is not None and k != "dataset"})
    if no_download:
        cfg["download"] = False
    run(cfg)


if __name__ == "__main__":
    main()
