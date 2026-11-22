import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol_key(result):
    cfg = result["config"]
    keys = ("dataset", "model", "profile", "comparison", "epochs", "batch_size", "lr", "momentum",
            "weight_decay", "milestones", "gamma", "split_seed", "omit", "selection", "warmup",
            "train_limit", "eval_limit", "deterministic")
    payload = {k: cfg[k] for k in keys}
    if cfg.get("mnist_shift"):
        payload["mnist_shift"] = cfg["mnist_shift"]
    payload.update(train_hash=result["data"]["train_hash"], val_hash=result["data"]["val_hash"],
                   test_hash=result["data"]["test_hash"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def summarize(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / "summary"
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("result.json"))]
    results = [result for result in results if result["test"] is not None]
    if not results:
        raise ValueError(f"No completed experiments under {root}")
    output.mkdir(parents=True, exist_ok=True)
    rows, groups, comparisons = [], {}, {}
    for result in results:
        cfg, test = result["config"], result["test"]
        protocol = protocol_key(result)
        settings = tuple(cfg[k] for k in ("temperature", "alpha", "beta", "ce_weight", "distill_weight"))
        key = (protocol, cfg["variant"], settings)
        groups.setdefault(key, []).append(result)
        comparisons.setdefault((cfg["dataset"], cfg["profile"], cfg["comparison"], protocol), []).append(result)
        rows.append(dict(dataset=cfg["dataset"], profile=cfg["profile"], comparison=cfg["comparison"],
                         protocol=protocol, variant=cfg["variant"], seed=cfg["seed"],
                         accuracy=test["accuracy"], validation=result["validation"]["accuracy"],
                         omitted_recall=test.get("omitted_recall"), other_accuracy=test.get("other_accuracy"),
                         parameters=result["complexity"]["parameters"], flops=result["complexity"]["flops"],
                         seconds=result["training_seconds"], selected_epoch=result["selected_epoch"],
                         temperature=cfg["temperature"], alpha=cfg["alpha"], beta=cfg["beta"],
                         ce_weight=cfg["ce_weight"], distill_weight=cfg["distill_weight"],
                         teacher_sha=result["teacher"]["sha256"] if result["teacher"] else None))
    write_csv(output / "runs.csv", rows)
    aggregate = []
    for (protocol, variant, settings), members in groups.items():
        cfg = members[0]["config"]
        seeds = [r["config"]["seed"] for r in members]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed results for {variant}; summarize one experiment root")
        values = [r["test"]["accuracy"] for r in members]
        aggregate.append(dict(dataset=cfg["dataset"], profile=cfg["profile"], comparison=cfg["comparison"],
                              protocol=protocol, variant=variant, seeds=" ".join(map(str, seeds)),
                              n=len(values), mean=statistics.mean(values),
                              std=statistics.stdev(values) if len(values) > 1 else None,
                              note="mean +/- sample SD" if len(values) > 1 else "single-seed exploratory"))
    write_csv(output / "aggregate.csv", aggregate)
    pairs = []
    for (dataset, profile, comparison, protocol), members in comparisons.items():
        if comparison != "cifar10_reversed":
            continue
        baselines = {r["config"]["seed"]: r for r in members if r["config"]["method"] == "ce"}
        for result in members:
            cfg = result["config"]
            if cfg["method"] != "kd" or cfg["seed"] not in baselines:
                continue
            baseline = baselines[cfg["seed"]]
            if result["initial_hash"] != baseline["initial_hash"] or cfg["ce_weight"] != baseline["config"]["ce_weight"]:
                raise ValueError("Invalid paired CIFAR-10 comparison")
            pairs.append(dict(profile=profile, protocol=protocol, seed=cfg["seed"],
                              ce=baseline["test"]["accuracy"], kd=result["test"]["accuracy"],
                              delta_pp=100 * (result["test"]["accuracy"] - baseline["test"]["accuracy"])))
    write_csv(output / "paired_cifar10.csv", pairs)
    pair_groups = {}
    for pair in pairs:
        pair_groups.setdefault((pair["profile"], pair["protocol"]), []).append(pair["delta_pp"])
    write_csv(output / "paired_cifar10_summary.csv", [
        dict(profile=profile, protocol=protocol, n=len(values), mean_delta_pp=statistics.mean(values),
             std_delta_pp=statistics.stdev(values) if len(values) > 1 else None)
        for (profile, protocol), values in pair_groups.items()])
    for (dataset, profile, comparison, protocol), members in comparisons.items():
        if members[0]["config"]["role"] == "teacher":
            continue
        stem = f"{profile}_{comparison}_{protocol}"
        fig, axis = plt.subplots(figsize=(max(6, len(members) * 0.5), 4))
        grouped = [r for r in aggregate if r["protocol"] == protocol]
        positions = np.arange(len(grouped))
        axis.bar(positions, [100*r["mean"] for r in grouped],
                 yerr=[100*(r["std"] or 0) for r in grouped], capsize=3)
        axis.set_xticks(positions, [r["variant"] for r in grouped], rotation=45, ha="right")
        axis.set(ylabel="Test accuracy (%)", title=f"{comparison} [{profile}]")
        fig.tight_layout()
        fig.savefig(output / f"{stem}.png", dpi=160)
        plt.close(fig)
        if comparison == "mnist_temperature":
            ordered = sorted(members, key=lambda r: r["config"]["temperature"])
            fig, axis = plt.subplots(figsize=(6, 4))
            for split in ("validation", "test"):
                axis.plot([r["config"]["temperature"] for r in ordered],
                          [r[split]["accuracy"]*100 for r in ordered], "o-", label=split)
            axis.set(xlabel="Temperature", ylabel="Accuracy (%)", title=f"Temperature study [{profile}]")
            axis.set_xticks(sorted({r["config"]["temperature"] for r in ordered}))
            axis.legend()
            fig.tight_layout()
            fig.savefig(output / f"{stem}_temperature.png", dpi=160)
            plt.close(fig)
        if members[0]["config"]["omit"] is not None:
            fig, axis = plt.subplots(figsize=(7, 5))
            for result in members:
                axis.scatter(result["test"]["other_accuracy"]*100, result["test"]["omitted_recall"]*100,
                             label=result["config"]["variant"], s=45,
                             marker="x" if result["config"]["ce_weight"] == 0 else "o")
                cm = np.asarray(result["test"]["confusion_matrix"])
                cm_fig, cm_axis = plt.subplots(figsize=(5, 4))
                cm_axis.imshow(cm, cmap="Blues")
                cm_axis.set(xlabel="Predicted", ylabel="True", xticks=range(10), yticks=range(10),
                            title=f"{result['config']['variant']} [{profile}]")
                cm_fig.tight_layout()
                cm_fig.savefig(output / f"{stem}_{result['config']['variant']}_seed{result['config']['seed']}_confusion.png", dpi=160)
                plt.close(cm_fig)
            axis.set(xlabel="Other nine classes accuracy (%)", ylabel="Omitted digit recall (%)",
                     title=f"Omitted-class tradeoff [{profile}]")
            axis.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
            fig.tight_layout()
            fig.savefig(output / f"{stem}_tradeoff.png", dpi=160)
            plt.close(fig)
    print(f"Wrote {len(results)} runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results and generate figures.")
    parser.add_argument("--root", default="runs/full")
    parser.add_argument("--output")
    args = parser.parse_args()
    summarize(args.root, args.output)
