"""Run the seven requested E/F experiments, validate pairing, and collect results."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import torch


def run_logged(command: list[str], project: Path, log: Path) -> None:
    """Stream a child process while saving its complete console output."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONPATH"] = str(project) + os.pathsep + environment.get("PYTHONPATH", "")
    with log.open("a", encoding="utf-8") as stream:
        with subprocess.Popen(
            command, cwd=project, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    tools = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    from models import make_model
    from train import file_hash, model_hash, seed_everything, validate_config

    root = project / "runs_public_teacher" / "full" / "cifar100"
    official_root = root.parent / "cifar100_kd_official"
    jobs = []
    for seed in (0, 1, 2):
        jobs.append((root / f"kd_seed{seed}", official_root / f"kd_official_seed{seed}",
                     {"ce_weight": 0.1, "distill_weight": 0.9}))
    for method in ("tckd", "nckd"):
        for seed in (1, 2):
            jobs.append((root / f"{method}_seed0", root / f"{method}_seed{seed}", {"seed": seed}))

    expected = []
    teacher_hashes = set()
    for original, destination, changes in jobs:
        source = json.loads((original / "result.json").read_text(encoding="utf-8"))
        cfg = source["config"] | changes | {"out": str(destination)}
        validate_config(cfg)
        if (cfg["profile"], cfg["split_seed"], cfg["epochs"], cfg["warmup"], cfg["temperature"]) != (
            "full", 661, 240, 20, 4.0
        ):
            raise ValueError(f"Unexpected original protocol in {original}")
        baseline = json.loads((root / f"ce_seed{cfg['seed']}" / "result.json").read_text(encoding="utf-8"))
        if str(torch.__version__) != baseline["environment"]["torch"]:
            raise ValueError("Use the original PyTorch environment")
        if torch.cuda.get_device_name(0) != baseline["environment"]["device"]:
            raise ValueError("Use the GPU used by the original runs")
        seed_everything(cfg["seed"], cfg["deterministic"])
        initial_hash = model_hash(make_model(cfg["model"], 100))
        if initial_hash != baseline["initial_hash"]:
            raise ValueError(f"Paired initialization changed for seed {cfg['seed']}")
        teacher_hash = file_hash(cfg["teacher"])
        if teacher_hash != source["teacher"]["sha256"]:
            raise ValueError("Teacher checkpoint changed")
        teacher_hashes.add(teacher_hash)
        expected.append((original, destination, changes, cfg, initial_hash))
        print(f"{destination.name}: {cfg['method']}, seed={cfg['seed']}, "
              f"CE={cfg['ce_weight']}, distill={cfg['distill_weight']}, "
              f"epochs={cfg['epochs']}, initial_hash={initial_hash}", flush=True)
    if len(teacher_hashes) != 1:
        raise ValueError("Expected the same teacher for every comparison")
    if not (tools / "eval_last.py").is_file():
        raise FileNotFoundError("Missing eval_last.py beside this script")
    confidence = tools / "cifar100_teacher_confidence"
    confidence_record = json.loads((confidence / "teacher_confidence.json").read_text(encoding="utf-8"))
    if confidence_record["teacher_sha256"] not in teacher_hashes:
        raise ValueError("B2 statistics used a different teacher")
    if confidence_record["temperature"] != 4.0 or confidence_record["one_minus_p_y"]["count"] != 45000:
        raise ValueError("B2 statistics must cover 45,000 training images at temperature 4")
    if args.plan_only:
        print("Preflight passed. Seven experiments planned. No training started.")
        return

    log = tools / "ef_training.log"
    for original, destination, changes, cfg, initial_hash in expected:
        command = [sys.executable, "-B", "-u", str(project / "train.py"),
                   "--config", str(original / "config.json"), "--out", str(destination)]
        for key, value in changes.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        run_logged(command, project, log)
        result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
        if result["config"] != cfg or result["initial_hash"] != initial_hash:
            raise ValueError(f"Completed experiment failed pairing/config checks: {destination}")
        reference = json.loads((original / "result.json").read_text(encoding="utf-8"))
        if result["data"] != reference["data"] or result["teacher"]["sha256"] not in teacher_hashes:
            raise ValueError(f"Completed experiment changed teacher or dataset: {destination}")
        if result["history"][-1]["epoch"] != 240:
            raise ValueError(f"Incomplete training: {destination}")

    for experiment_root in (root, official_root):
        run_logged([sys.executable, "-B", "-u", str(tools / "eval_last.py"),
                    "--root", str(experiment_root), "--data-root", str(project / "data"),
                    "--device", "cuda"], project, tools / f"{experiment_root.name}_last_epoch_eval.log")
        run_logged([sys.executable, "-B", str(project / "report.py"),
                    "--root", str(experiment_root)], project, log)

    stage = tools / "gitlab-upload" / "results" / "full"
    for experiment_root in (root, official_root):
        target_root = stage / experiment_root.name
        for result_path in experiment_root.glob("*/result.json"):
            target = target_root / result_path.parent.name
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("config.json", "history.csv", "result.json"):
                shutil.copy2(result_path.parent / filename, target / filename)
        target_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(experiment_root / "last_epoch_eval.csv", target_root / "last_epoch_eval.csv")
        shutil.copytree(experiment_root / "summary", target_root / "summary", dirs_exist_ok=True)

    evaluations = []
    for experiment_root in (root, official_root):
        with (experiment_root / "last_epoch_eval.csv").open(encoding="utf-8", newline="") as stream:
            evaluations.append({int(row["seed"]): float(row["last_epoch_test"])
                                for row in csv.DictReader(stream) if row["variant"] == "kd"})
    if set(evaluations[0]) != {0, 1, 2} or set(evaluations[1]) != {0, 1, 2}:
        raise ValueError("Expected all three seeds for both KD weighting settings")
    comparisons = [dict(seed=seed, original_kd=evaluations[0][seed],
                        official_weight_kd=evaluations[1][seed],
                        official_minus_original_pp=100 * (evaluations[1][seed] - evaluations[0][seed]))
                   for seed in (0, 1, 2)]
    with (stage / "cifar100_kd_official" / "official_kd_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    differences = [row["official_minus_original_pp"] for row in comparisons]
    print(f"Official-weight KD minus original KD at epoch 240: "
          f"{statistics.mean(differences):+.3f} +/- {statistics.stdev(differences):.3f} pp", flush=True)
    shutil.copytree(confidence, stage / "cifar100" / "teacher_confidence", dirs_exist_ok=True)
    for filename in ("cifar100_last_epoch_eval.log", "cifar100_kd_official_last_epoch_eval.log"):
        shutil.copy2(tools / filename, stage / filename.replace(".log", ".txt"))
    for filename in ("eval_last.py", "teacher_confidence.py", "run_ef.py"):
        shutil.copy2(tools / filename, tools / "gitlab-upload" / filename)
    archive_path = tools / "cifar_followup_results.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted((tools / "gitlab-upload").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tools / "gitlab-upload").as_posix())
    print(f"E/F finished. Collected results: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
