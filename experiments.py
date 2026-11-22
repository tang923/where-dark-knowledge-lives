import argparse
from pathlib import Path
import random

from train import default_config, run, write_json


class Experiments:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output) / args.profile
        self.omit = random.Random(args.omit_seed).randrange(10)

    def base(self, dataset):
        cfg = default_config(dataset)
        cfg.update(profile=self.args.profile, data_root=self.args.data_root,
                   device=self.args.device, workers=self.args.workers,
                   download=not self.args.no_download)
        if self.args.profile == "smoke":
            cfg.update(epochs=1, train_limit=200, eval_limit=100, milestones=[], warmup=0)
        elif self.args.profile == "pilot":
            cfg.update(epochs=3 if dataset == "mnist" else 2,
                       train_limit=5000 if dataset == "mnist" else 2000,
                       eval_limit=1000 if dataset == "mnist" else 500,
                       milestones=[], warmup=0)
        return cfg

    def launch(self, cfg, name):
        cfg = cfg | {"out": str(self.root / cfg["dataset"] / name)}
        if self.args.dry_run:
            print(f"{name}: {cfg['model']} {cfg['method']} seed={cfg['seed']} "
                  f"epochs={cfg['epochs']} T={cfg['temperature']} beta={cfg['beta']} "
                  f"CE={cfg['ce_weight']} omit={cfg['omit']}")
            return dict(config=cfg, validation={"accuracy": 0.0}, fingerprint="dry-run")
        return run(cfg)

    def teacher(self, dataset):
        if dataset == "cifar100" and self.args.cifar100_teacher:
            if not self.args.teacher_source:
                raise ValueError("--cifar100-teacher requires --teacher-source")
            return self.args.cifar100_teacher
        cfg = self.base(dataset)
        model = {"mnist": "mlp1200", "cifar10": "resnet20", "cifar100": "resnet56"}[dataset]
        cfg.update(model=model, role="teacher", comparison=f"{dataset}_teacher", variant="teacher", seed=0)
        self.launch(cfg, "teacher_seed0")
        return str(self.root / dataset / "teacher_seed0" / "best.pt")

    def student(self, dataset, teacher):
        cfg = self.base(dataset)
        cfg.update(model={"mnist": "mlp800", "cifar10": "resnet56", "cifar100": "resnet20"}[dataset],
                   teacher_model={"mnist": "mlp1200", "cifar10": "resnet20", "cifar100": "resnet56"}[dataset])
        if dataset == "cifar100" and self.args.cifar100_teacher:
            cfg.update(teacher_source=self.args.teacher_source,
                       teacher_training_data=self.args.teacher_training_data)
        else:
            cfg["teacher_training_data"] = "local stratified training split; validation excluded"
        cfg["teacher"] = teacher
        return cfg

    def method(self, base, method, name, **changes):
        cfg = base | dict(method=method, variant=name) | changes
        if method == "ce":
            cfg["teacher"] = None
        return self.launch(cfg, f"{name}_seed{cfg['seed']}")

    def lock(self, filename, obj):
        if not self.args.dry_run:
            write_json(self.root / "mnist" / filename, obj)

    def mnist_full(self):
        base = self.student("mnist", self.teacher("mnist"))
        base.update(seed=self.args.seeds[0], comparison="mnist_full")
        self.method(base, "ce", "ce_full")
        temperatures = (1, 2, 4, 8, 20)
        candidates = [self.method(base, "kd", f"kd_full_T{t}", temperature=float(t),
                                  comparison="mnist_temperature") for t in temperatures]
        best = max(candidates, key=lambda result: result["validation"]["accuracy"])
        temperature = 4.0 if self.args.dry_run else best["config"]["temperature"]
        self.lock("temperature_selection.json", dict(rule="highest full-transfer validation accuracy; first T breaks ties",
                  temperature=temperature, seed=base["seed"], omitted_digit=self.omit,
                  omit_seed=self.args.omit_seed, candidates=[dict(fingerprint=r["fingerprint"],
                  temperature=r["config"]["temperature"], validation=r["validation"]["accuracy"]) for r in candidates]))
        return base | dict(temperature=temperature)

    def omitted_base(self, base):
        return base | dict(omit=self.omit, selection="last", comparison=f"mnist_omitted_{self.omit}")

    def mnist(self):
        base = self.omitted_base(self.mnist_full())
        self.method(base, "ce", f"ce_omit{self.omit}")
        self.method(base, "kd", f"kd_omit{self.omit}")

    def cifar(self, dataset):
        base = self.student(dataset, self.teacher(dataset))
        base["comparison"] = "cifar10_reversed" if dataset == "cifar10" else "cifar100_components"
        for i, seed in enumerate(self.args.seeds):
            methods = ("ce", "kd") if dataset == "cifar10" else (
                ("ce", "kd", "tckd", "nckd", "dkd") if i == 0 else ("ce", "kd", "dkd"))
            for method in methods:
                self.method(base | dict(seed=seed), method, method)

    def extension(self):
        base = self.mnist_full()
        betas = (1.0, 2.0, 4.0, 8.0)
        candidates = [self.method(base, "dkd", f"dkd_full_b{b:g}", beta=b) for b in betas]
        best = max(candidates, key=lambda r: r["validation"]["accuracy"])
        beta = 2.0 if self.args.dry_run else best["config"]["beta"]
        self.lock("extension_selection.json", dict(temperature=base["temperature"], alpha=base["alpha"],
                  beta=beta, omitted_digit=self.omit, omit_seed=self.args.omit_seed,
                  ce_weight=base["ce_weight"], distill_weight=base["distill_weight"],
                  selection="final epoch on omitted transfer; no test-based tuning",
                  candidates=[dict(beta=r["config"]["beta"], fingerprint=r["fingerprint"],
                                   validation=r["validation"]["accuracy"]) for r in candidates]))
        self.method(base, "tckd", "tckd_full")
        self.method(base, "nckd", f"nckd_full_b{beta:g}", beta=beta)
        omitted = self.omitted_base(base)
        for method in ("ce", "kd", "tckd"):
            self.method(omitted, method, f"{method}_omit{self.omit}")
        self.method(omitted, "nckd", f"nckd_omit{self.omit}_b{beta:g}", beta=beta)
        for b in betas:
            self.method(omitted, "dkd", f"dkd_omit{self.omit}_b{b:g}", beta=b)
        self.method(omitted, "kd", f"kd_noce_omit{self.omit}", ce_weight=0.0)
        for b in betas:
            self.method(omitted, "dkd", f"dkd_noce_omit{self.omit}_b{b:g}", beta=b, ce_weight=0.0)


def main():
    parser = argparse.ArgumentParser(description="Run the planned experiments. Use --dry-run to preview the run matrix.")
    parser.add_argument("--track", choices=["mnist", "cifar10", "cifar100", "extension", "all"], required=True)
    parser.add_argument("--profile", choices=["smoke", "pilot", "full"], default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="CIFAR: later seeds repeat principal comparisons only")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--omit-seed", type=int, default=661)
    parser.add_argument("--cifar100-teacher", help="Optional compatible public teacher checkpoint")
    parser.add_argument("--teacher-source", help="Public checkpoint URL / attribution")
    parser.add_argument("--teacher-training-data", default="unknown; may include our held-out validation images")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("Use distinct nonnegative seeds")
    suite = Experiments(args)
    if args.dry_run:
        print("Plan only. MNIST T/beta below use placeholders; real runs select by full-transfer validation.")
    tracks = ["mnist", "cifar10", "cifar100", "extension"] if args.track == "all" else [args.track]
    for track in tracks:
        if track.startswith("cifar"):
            suite.cifar(track)
        else:
            getattr(suite, track)()


if __name__ == "__main__":
    main()
