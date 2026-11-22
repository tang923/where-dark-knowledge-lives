"""Run the MNIST follow-up experiments from the original full Colab configurations."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from data import make_datasets, make_loader
from train import evaluate, file_hash, load_teacher, run, write_json


DIGITS = (2, 3, 7, 9)
SEEDS = (0, 1, 2)
TEMPERATURES = (1, 2, 4, 8, 20)
BETAS = (1, 2, 4, 8)
OMITTED_VARIANTS = ("ce_omit2", "tckd_omit2", "kd_omit2", "nckd_omit2_b1",
                    "dkd_omit2_b1", "kd_noce_omit2", "dkd_noce_omit2_b1")
WEAK_VARIANTS = ("ce_full", "kd_full_T2", "tckd_full", "nckd_full_b1", "dkd_full_b1")
VALIDATION_TARGET = 0.99
FIXED_TEACHER = "teacher_shift2_seed0"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def original_config(source, variant, data_root=None, workers=None):
    cfg = read_json(source / f"{variant}_seed0" / "config.json")["config"]
    expected = dict(dataset="mnist", profile="full", seed=0, split_seed=661,
                    epochs=20, lr=0.01, batch_size=128)
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError(f"Expected original MNIST full configuration for {variant}")
    if data_root is not None:
        cfg["data_root"] = str(data_root)
    if workers is not None:
        cfg["workers"] = workers
    return cfg


def teacher_path(source):
    return str((source / "teacher_seed0" / "best.pt").resolve())


def omitted_config(source, output, variant, digit, seed, data_root=None, workers=None):
    cfg = original_config(source, variant, data_root, workers)
    if cfg["omit"] != 2 or cfg["selection"] != "last":
        raise ValueError(f"Expected the original omitted-digit configuration for {variant}")
    name = variant.replace("omit2", f"omit{digit}")
    return cfg | dict(omit=digit, seed=seed, temperature=2.0, alpha=1.0, beta=1.0,
                      teacher=None if cfg["method"] == "ce" else teacher_path(source),
                      comparison=f"mnist_omit_followup_{digit}", variant=name,
                      out=str(output / f"{name}_seed{seed}"))


def check_pair(results):
    for key in ("initial_hash", "train_hash", "val_hash", "test_hash"):
        values = {r["initial_hash"] if key == "initial_hash" else r["data"][key] for r in results}
        if len(values) != 1:
            raise ValueError(f"Paired students differ in {key}")


def run_omitted(source, output, digits, dry_run, data_root=None, workers=None):
    for digit in digits:
        for seed in SEEDS:
            configs = [omitted_config(source, output, variant, digit, seed, data_root, workers)
                       for variant in OMITTED_VARIANTS]
            if dry_run:
                for cfg in configs:
                    print(cfg["out"], cfg["comparison"], cfg["method"], f"CE={cfg['ce_weight']}")
                continue
            check_pair([run(cfg) for cfg in configs])


def summarize_omitted(source, output, *, partial=False):
    complete_groups = {digit: {} for digit in DIGITS}
    missing = []
    for digit in DIGITS:
        for seed in SEEDS:
            paths = {variant: output / f"{variant.replace('omit2', f'omit{digit}')}_seed{seed}" / "result.json"
                     for variant in OMITTED_VARIANTS}
            missing_paths = [path for path in paths.values() if not path.is_file()]
            if missing_paths:
                missing.extend(str(path.relative_to(output)) for path in missing_paths)
            else:
                complete_groups[digit][seed] = paths
    expected = len(DIGITS) * len(SEEDS) * len(OMITTED_VARIANTS)
    print(f"Completed runs: {expected - len(missing)}/{expected}", flush=True)
    if missing and not partial:
        raise SystemExit(f"Missing {len(missing)} results, starting with {missing[0]}. "
                         "Use omitted-summary --partial to summarize completed pairs.")
    destination = output / "partial_summary" if partial else output
    destination.mkdir(parents=True, exist_ok=True)
    grouped = {variant: [] for variant in OMITTED_VARIANTS}
    per_digit = {}
    reproduction = {}
    curve_keys = ("omitted_recall", "omitted_mean_probability", "omitted_mean_logit", "omitted_mean_margin")
    for digit, groups in complete_groups.items():
        if not groups:
            continue
        per_digit[str(digit)] = {}
        histories = {variant: [] for variant in OMITTED_VARIANTS}
        for seed, paths in groups.items():
            results = []
            for variant, path in paths.items():
                cfg = omitted_config(source, output, variant, digit, seed)
                result = read_json(path)
                if (len(result["history"]) != cfg["epochs"] or
                        any(row.get(key) is None for row in result["history"] for key in curve_keys)):
                    raise ValueError(f"Incomplete validation curves in {cfg['out']}")
                results.append(result)
                histories[variant].append(result["history"])
                grouped[variant].append(result["test"])
                per_digit[str(digit)].setdefault(variant, []).append(result["test"])
                if digit == 2 and seed == 0:
                    old = read_json(source / f"{variant}_seed0" / "result.json")
                    reproduction[variant] = dict(
                        same_initial_hash=old["initial_hash"] == result["initial_hash"],
                        same_teacher=(old["teacher"] or {}).get("sha256") ==
                                     (result["teacher"] or {}).get("sha256"),
                        same_data=old["data"] == result["data"],
                        same_test_confusion=old["test"]["confusion_matrix"] == result["test"]["confusion_matrix"],
                        old_recall=old["test"]["omitted_recall"], new_recall=result["test"]["omitted_recall"],
                        old_environment=old["environment"], new_environment=result["environment"])
            check_pair(results)
        plot_omitted_curves(digit, histories, curve_keys, destination)

    def metrics(group):
        summary = {}
        for name, rows in group.items():
            summary[name] = {}
            for key in ("accuracy", "omitted_recall", "other_accuracy"):
                values = [row[key] for row in rows]
                summary[name][key] = dict(mean=statistics.mean(values),
                                          std=statistics.stdev(values) if len(values) > 1 else None)
        return summary

    summary_path = destination / "omitted_summary.json"
    write_json(summary_path, dict(
        completed_runs=expected - len(missing), expected_runs=expected, missing_results=missing,
        paired_seeds_by_digit={str(digit): list(groups) for digit, groups in complete_groups.items()},
        per_digit={digit: metrics(group) for digit, group in per_digit.items()},
        across_digit_seed_mean={name: {key: statistics.mean(row[key] for row in rows)
                                     for key in ("accuracy", "omitted_recall", "other_accuracy")}
                                 for name, rows in grouped.items()} if not missing else None,
        digit2_seed0_reproduction=reproduction))
    print(f"Wrote {summary_path}")


def plot_omitted_curves(digit, histories, keys, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    labels = ("Recall", "Mean probability (T=1)", "Mean logit", "Mean margin")
    methods = ("CE", "CE + TCKD", "CE + KD", "CE + NCKD", "CE + DKD", "KD", "DKD")
    for axis, key, label in zip(axes.flat, keys, labels):
        for method, runs in zip(methods, histories.values()):
            rows = list(zip(*runs))
            axis.plot([epoch[0]["epoch"] for epoch in rows],
                      [statistics.mean(row[key] for row in epoch) for epoch in rows], label=method)
        axis.set(xlabel="Epoch", ylabel=label)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4)
    n = len(next(iter(histories.values())))
    seeds = "1 seed" if n == 1 else f"mean of {n} seeds"
    fig.suptitle(f"Validation on omitted digit {digit} ({seeds})")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(output / f"validation_omit{digit}.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def teacher_stats(source, checkpoint, model_name, output, data_root=None, workers=None):
    cfg = original_config(source, "teacher", data_root, workers) | dict(
        teacher=str(checkpoint), teacher_model=model_name, mnist_shift=0)
    # One pass over the original training split; masking removes the selected digit for each summary.
    (train_set, _, _), info = make_datasets(cfg)
    device = torch.device(cfg["device"])
    teacher, _ = load_teacher(cfg, 10, device)
    all_logits, all_targets = [], []
    for images, targets in make_loader(train_set, cfg):
        all_logits.append((teacher(images.to(device)) / 2.0).cpu())
        all_targets.append(targets)
    logits, y = torch.cat(all_logits), torch.cat(all_targets)
    mask = torch.nn.functional.one_hot(y, 10).bool()
    non_target_logits = logits.masked_fill(mask, -torch.inf)
    conditional = non_target_logits.softmax(1)
    # Sum the nine non-target probabilities without subtracting from a rounded p_y.
    weight = logits.softmax(1).masked_fill(mask, 0).sum(1)
    summaries = {}
    for digit in DIGITS:
        keep = y != digit
        weights, signals = weight[keep], conditional[keep, digit]
        p10, median, p90 = weights.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
        summaries[str(digit)] = dict(
            count=int(keep.sum()),
            kd_nckd_weight=dict(mean=weights.mean().item(), median=median, p10=p10, p90=p90),
            mean_conditional_omitted_probability=signals.mean().item(),
            by_true_digit={str(c): dict(count=int((y == c).sum()),
                                       mean=conditional[y == c, digit].mean().item())
                           for c in range(10) if c != digit})
    write_json(output, dict(teacher_model=model_name, teacher_sha256=file_hash(checkpoint),
                            temperature=2.0, dkd_beta=1.0, split_seed=cfg["split_seed"],
                            full_train_hash=info["train_hash"], omitted_digits=summaries))
    print(f"Wrote {output}")


def strong_teacher(source, output, data_root=None, workers=None):
    cfg = original_config(source, "teacher", data_root, workers) | dict(
        model="mlp1200d", epochs=60, milestones=[30, 45], comparison="mnist_strong_teacher",
        test_validation_floor=VALIDATION_TARGET)
    for name, shift in (("teacher_seed0", 0), ("teacher_shift2_seed0", 2)):
        result = run(cfg | dict(mnist_shift=shift, out=str(output / name)))
        if result["validation"]["accuracy"] >= VALIDATION_TARGET:
            write_json(output / "teacher_selection.json", dict(
                directory=name, validation=result["validation"]["accuracy"], target=VALIDATION_TARGET,
                fingerprint=result["fingerprint"], sha256=file_hash(output / name / "best.pt")))
            return
    raise SystemExit("Neither recipe reached 99% validation accuracy. Strong distillation is blocked; "
                     "weak-students can run independently.")


def selected_strong_teacher(output):
    selection_path = output / "teacher_selection.json"
    if not selection_path.is_file():
        raise SystemExit("No strong teacher met the 99% validation target. Run strong-teacher first.")
    selection = read_json(selection_path)
    folder = output / selection["directory"]
    result = read_json(folder / "result.json")
    checkpoint = folder / "best.pt"
    if (result["validation"]["accuracy"] < VALIDATION_TARGET or
            result["fingerprint"] != selection["fingerprint"] or
            file_hash(checkpoint) != selection["sha256"]):
        raise ValueError("The selected strong teacher failed validation or changed")
    return str(checkpoint.resolve())


def full_student(source, output, template, variant, seed, data_root, workers, checkpoint=None, **changes):
    cfg = original_config(source, template, data_root, workers)
    if cfg["method"] != "ce":
        cfg.update(teacher=checkpoint or teacher_path(source),
                   teacher_model="mlp1200d" if checkpoint else "mlp1200")
    group = "strong" if checkpoint else "weak"
    comparison = "mnist_strong_temperature" if checkpoint and cfg["method"] == "kd" else f"mnist_{group}_full"
    return run(cfg | dict(comparison=comparison, variant=variant, seed=seed,
                          out=str(output / group / f"{variant}_seed{seed}")) | changes)


def accuracy_deltas(ce, students):
    return {r["config"]["method"]: 100 * (r["test"]["accuracy"] - ce["test"]["accuracy"])
            for r in students}


def summarize_deltas(deltas):
    return {method: dict(mean_delta_pp=statistics.mean(values), std_delta_pp=statistics.stdev(values))
            for method, values in deltas.items()}


def run_weak_students(source, output, data_root=None, workers=None):
    results_by_seed, paired = {}, []
    for seed in SEEDS:
        results = [read_json(source / f"{name}_seed0" / "result.json") if seed == 0 else
                   full_student(source, output, name, name, seed, data_root, workers)
                   for name in WEAK_VARIANTS]
        check_pair(results)
        results_by_seed[seed] = results
        ce, *students = results
        paired.append(dict(seed=seed, ce=ce["test"]["accuracy"], delta_pp=accuracy_deltas(ce, students)))
    deltas = {method: [pair["delta_pp"][method] for pair in paired] for method in paired[0]["delta_pp"]}
    path = output / "paired_weak_comparison.json"
    write_json(path, dict(per_seed=paired, methods=summarize_deltas(deltas)))
    print(f"Wrote {path}")
    return results_by_seed


def run_students(source, output, data_root=None, workers=None):
    checkpoint = selected_strong_teacher(output)

    def student(template, variant, seed, **changes):
        return full_student(source, output, template, variant, seed, data_root, workers, checkpoint, **changes)

    kd = {t: [student(f"kd_full_T{t}", f"kd_full_T{t}", seed) for seed in SEEDS]
          for t in TEMPERATURES}
    val_t = {t: statistics.mean(r["validation"]["accuracy"] for r in results) for t, results in kd.items()}
    temperature = max(val_t, key=val_t.get)
    sweep = {b: student(f"dkd_full_b{b}", f"dkd_full_b{b}", 0, temperature=float(temperature))
             for b in BETAS}
    val_b = {b: r["validation"]["accuracy"] for b, r in sweep.items()}
    beta = max(val_b, key=val_b.get)
    write_json(output / "student_selection.json", dict(temperature=temperature, beta=beta,
                                                        validation_by_t=val_t, validation_by_beta=val_b))
    weak_results = run_weak_students(source, output, data_root, workers)
    paired = []
    for seed in SEEDS:
        ce, *weak = weak_results[seed]
        settings = dict(temperature=float(temperature), beta=float(beta), alpha=1.0)
        nckd = f"nckd_full_b{beta}"
        dkd = f"dkd_full_b{beta}"
        strong = [kd[temperature][seed],
                  student("tckd_full", "tckd_full", seed, **settings),
                  student("nckd_full_b1", nckd, seed, **settings),
                  sweep[beta] if seed == 0 else student(dkd, dkd, seed, **settings)]
        check_pair([ce, *weak, *strong])
        deltas = {group: accuracy_deltas(ce, results)
                  for group, results in (("weak", weak), ("strong", strong))}
        paired.append(dict(seed=seed, ce=ce["test"]["accuracy"], delta_pp=deltas))
    kd_summary = summarize_deltas({group: [pair["delta_pp"][group]["kd"] for pair in paired]
                                  for group in ("weak", "strong")})
    write_json(output / "paired_teacher_comparison.json", dict(per_seed=paired, kd=kd_summary))


def fixed_teacher_test(source, output, data_root=None, workers=None):
    folder = output / FIXED_TEACHER
    checkpoint = (folder / "best.pt").resolve()
    candidate = read_json(folder / "result.json")
    checkpoint_sha = file_hash(checkpoint)
    path = output / "fixed_kd" / "teacher_evaluation.json"
    if path.is_file():
        record = read_json(path)
        if (record["teacher_sha256"] != checkpoint_sha or
                record["teacher_fingerprint"] != candidate["fingerprint"]):
            raise ValueError("The fixed teacher changed after its test evaluation")
        print(f"[cached] {path}")
        return record

    dropout = read_json(output / "teacher_seed0" / "result.json")
    validation = candidate["validation"]["accuracy"]
    if validation < dropout["validation"]["accuracy"]:
        raise ValueError("The fixed candidate is no longer the best by validation accuracy")
    cfg = original_config(source, "teacher", data_root, workers) | dict(
        teacher=str(checkpoint), teacher_model="mlp1200d")
    (_, _, test_set), info = make_datasets(cfg)
    if info != candidate["data"]:
        raise ValueError("The fixed teacher's data split changed")
    device = torch.device(cfg["device"])
    teacher, _ = load_teacher(cfg, 10, device)
    test = evaluate(teacher, make_loader(test_set, cfg), device, 10)
    record = dict(teacher=FIXED_TEACHER, teacher_sha256=checkpoint_sha,
                  teacher_fingerprint=candidate["fingerprint"], selected_epoch=candidate["selected_epoch"],
                  selection="Higher validation accuracy of the two completed teacher recipes",
                  validation_by_recipe={FIXED_TEACHER: validation,
                                        "teacher_seed0": dropout["validation"]["accuracy"]},
                  temperature=2.0, data=info, test=test,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                   device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"))
    write_json(path, record)
    print(f"Fixed teacher: val={validation:.4%}, test={test['accuracy']:.4%}. Wrote {path}")
    return record


def run_fixed_kd(source, output, data_root=None, workers=None):
    fixed = output / "fixed_kd"
    cfg = original_config(source, "kd_full_T2", data_root, workers) | dict(
        teacher=str((output / FIXED_TEACHER / "best.pt").resolve()), teacher_model="mlp1200d",
        temperature=2.0, comparison="mnist_fixed_teacher_kd")
    baseline_paths = [(source if seed == 0 else output / "weak") / f"ce_full_seed{seed}" / "result.json"
                      for seed in SEEDS]
    baselines = [read_json(path) for path in baseline_paths]
    teacher = fixed_teacher_test(source, output, data_root, workers)
    data_keys = ("train_hash", "val_hash", "test_hash")
    for seed, ce, baseline_path in zip(SEEDS, baselines, baseline_paths):
        baseline_cfg = ce["config"]
        if (baseline_cfg["seed"] != seed or baseline_cfg["method"] != "ce" or
                any(baseline_cfg[key] != cfg[key] for key in ("dataset", "model", "profile", "omit",
                                                              "split_seed", "epochs", "batch_size", "lr",
                                                              "milestones", "selection", "ce_weight")) or
                any(ce["data"][key] != teacher["data"][key] for key in data_keys)):
            raise ValueError(f"Invalid existing CE baseline: {baseline_path}")

    paired = []
    for seed, ce, baseline_path in zip(SEEDS, baselines, baseline_paths):
        result = run(cfg | dict(seed=seed, out=str(fixed / f"kd_full_T2_seed{seed}")))
        if (result["teacher"]["sha256"] != teacher["teacher_sha256"] or
                any(ce["data"][key] != result["data"][key] for key in data_keys)):
            raise ValueError(f"The KD run does not match the fixed teacher or CE data for seed {seed}")
        init_match = ce["initial_hash"] == result["initial_hash"]
        if seed == 0 and not init_match:
            print("Seed 0: same seed and data, different initialization hash from the original Colab CE run.")
        elif not init_match:
            raise ValueError(f"The local KD initialization does not match CE for seed {seed}")
        ce_acc, kd_acc = ce["test"]["accuracy"], result["test"]["accuracy"]
        paired.append(dict(seed=seed, ce=ce_acc, kd=kd_acc, delta_pp=100 * (kd_acc - ce_acc),
                           initial_hash_match=init_match, ce_result=str(baseline_path),
                           ce_environment=ce["environment"], kd_environment=result["environment"]))

    deltas = dict(all_seeds=[pair["delta_pp"] for pair in paired],
                  matching_initialization=[pair["delta_pp"] for pair in paired if pair["initial_hash_match"]])
    summary = fixed / "paired_kd_vs_ce.json"
    write_json(summary, dict(teacher_sha256=teacher["teacher_sha256"], temperature=2.0,
                            per_seed=paired, delta_summary=summarize_deltas(deltas)))
    print(f"Wrote {summary}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("omitted", "omitted-summary", "teacher-stats", "strong-teacher",
                                        "strong-students", "weak-students", "fixed-teacher-test", "fixed-kd"))
    parser.add_argument("--root", type=Path, default=Path("runs/full"))
    parser.add_argument("--source", type=Path, help="Original full MNIST run directory; defaults to ROOT/mnist")
    parser.add_argument("--data-root", type=Path, help="Override the data path saved in the original configurations")
    parser.add_argument("--workers", type=int, help="Override data-loading processes (0 avoids Windows process startup)")
    parser.add_argument("--digits", nargs="+", type=int, choices=DIGITS, default=DIGITS)
    parser.add_argument("--dry-run", action="store_true", help="Print the omitted-digit run plan")
    parser.add_argument("--partial", action="store_true", help="Summarize complete pairs before all runs finish")
    parser.add_argument("--strong", action="store_true", help="Use the selected strong teacher for teacher-stats")
    args = parser.parse_args()
    source = args.source or args.root / "mnist"
    omitted = args.root / "mnist_omit"
    strong = args.root / "mnist_strong"
    if args.dry_run and args.task != "omitted":
        parser.error("--dry-run is only supported for omitted")
    if args.partial and args.task != "omitted-summary":
        parser.error("--partial is only supported for omitted-summary")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.task == "omitted":
        run_omitted(source, omitted, dict.fromkeys(args.digits), args.dry_run, args.data_root, args.workers)
    elif args.task == "omitted-summary":
        summarize_omitted(source, omitted, partial=args.partial)
    elif args.task == "teacher-stats":
        checkpoint = selected_strong_teacher(strong) if args.strong else teacher_path(source)
        model_name = "mlp1200d" if args.strong else "mlp1200"
        output = args.root / "mnist_teacher_stats" / ("strong.json" if args.strong else "original.json")
        teacher_stats(source, checkpoint, model_name, output, args.data_root, args.workers)
    elif args.task == "strong-teacher":
        strong_teacher(source, strong, args.data_root, args.workers)
    elif args.task == "weak-students":
        run_weak_students(source, strong, args.data_root, args.workers)
    elif args.task == "fixed-teacher-test":
        fixed_teacher_test(source, strong, args.data_root, args.workers)
    elif args.task == "fixed-kd":
        run_fixed_kd(source, strong, args.data_root, args.workers)
    else:
        run_students(source, strong, args.data_root, args.workers)


if __name__ == "__main__":
    main()
