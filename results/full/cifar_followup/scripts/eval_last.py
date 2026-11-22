"""Evaluate the final-epoch checkpoint (last.pt) of every run on the test set.

Purpose: CIFAR-100 runs select best.pt with a validation split that the public
teacher was trained on. last.pt (epoch 240) is chosen without any validation
data, so comparing the two checks whether that selection biased the results.

Run from the repository root (needs models.py, data.py, train.py):

    python eval_last.py --root runs_public_teacher/full/cifar100 --data-root data
    python eval_last.py --root runs/full/cifar10 --data-root data   # optional

Writes <root>/last_epoch_eval.csv and prints a summary.
"""

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import torch

from data import make_datasets, make_loader
from models import make_model
from train import evaluate, seed_everything

PAIRS = [("kd", "ce"), ("dkd", "ce"), ("dkd", "kd"), ("nckd", "ce"), ("tckd", "ce")]


def paired(a, b):
    """Paired differences in points, mean, sample SD, t, and two-sided p."""
    seeds = sorted(set(a) & set(b))
    d = [100 * (a[s] - b[s]) for s in seeds]
    if len(d) < 2:
        return seeds, d, None
    mean, sd = st.mean(d), st.stdev(d)
    if sd == 0:
        # Identical zero differences use the explicit t=0, p=1 convention.
        t, p = (0.0, 1.0) if mean == 0 else (math.copysign(float("inf"), mean), 0.0)
        return seeds, d, (mean, sd, t, p)
    t = mean / (sd / math.sqrt(len(d)))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), len(d) - 1)
    except ImportError:
        p = 1 - abs(t) / math.sqrt(t * t + 2) if len(d) == 3 else float("nan")
    return seeds, d, (mean, sd, t, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="folder containing <variant>_seed<k>/ run folders")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(args.root)
    runs = sorted(p.parent for p in root.glob("*/last.pt"))
    if not runs:
        raise SystemExit(f"No */last.pt under {root}")

    rows, test_loader, loader_key = [], None, None
    for run in runs:
        ckpt = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        cfg = dict(ckpt["config"])
        if ckpt["epoch"] != cfg["epochs"]:
            print(f"[skip] {run.name}: last.pt is from epoch {ckpt['epoch']}/{cfg['epochs']} (unfinished)")
            continue
        cfg.update(data_root=args.data_root, download=False, device=args.device, workers=0)
        seed_everything(cfg["seed"], cfg.get("deterministic", True))
        classes = 100 if cfg["dataset"] == "cifar100" else 10

        key = (cfg["dataset"], cfg["split_seed"], cfg["omit"], cfg["eval_limit"])
        if key != loader_key:
            (_, _, test_set), _ = make_datasets(cfg)
            test_loader, loader_key = make_loader(test_set, cfg), key

        model = make_model(cfg["model"], classes).to(args.device)
        model.load_state_dict(ckpt["model"])
        last = evaluate(model, test_loader, torch.device(args.device), classes)

        result_path = run / "result.json"
        best = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        best_test = best.get("test", {}).get("accuracy")
        variant = cfg.get("variant") or cfg["method"]
        row = dict(run=run.name, variant=variant, seed=cfg["seed"],
                   selected_epoch=best.get("selected_epoch"),
                   best_val_ckpt_test=best_test,
                   last_epoch_test=last["accuracy"],
                   last_epoch_val=ckpt["history"][-1]["val_accuracy"],
                   diff_pp=None if best_test is None else round(100 * (last["accuracy"] - best_test), 2))
        rows.append(row)
        print(f"{run.name:16s} best-val ckpt (ep {row['selected_epoch']}): {best_test}  "
              f"last epoch: {last['accuracy']:.4f}  diff {row['diff_pp']} pp", flush=True)

    if not rows:
        raise SystemExit(f"No completed checkpoints under {root}")
    out = root / "last_epoch_eval.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Summary: test accuracy (%) ===")
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["seed"]] = r
    for variant, seeds in sorted(by.items()):
        best = [100 * r["best_val_ckpt_test"] for r in seeds.values() if r["best_val_ckpt_test"] is not None]
        last = [100 * r["last_epoch_test"] for r in seeds.values()]
        fmt = lambda xs: f"{st.mean(xs):.2f}" + (f" +- {st.stdev(xs):.2f}" if len(xs) > 1 else "")
        print(f"{variant:6s} n={len(last)}  best-val ckpt: {fmt(best) if best else 'n/a':15s}  last epoch: {fmt(last)}")

    print("\n=== Paired differences, last epoch (pp) ===")
    for a, b in PAIRS:
        if a in by and b in by:
            acc = lambda v: {s: r["last_epoch_test"] for s, r in by[v].items()}
            seeds, d, stats = paired(acc(a), acc(b))
            if stats:
                mean, sd, t, p = stats
                print(f"{a}-{b}: seeds {seeds} diffs {[round(x, 2) for x in d]}  "
                      f"mean {mean:+.2f} +- {sd:.2f}  t={t:.2f}  p={p:.4f}")
            else:
                print(f"{a}-{b}: seeds {seeds} diffs {[round(x, 2) for x in d]} (single seed)")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
