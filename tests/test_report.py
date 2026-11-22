"""检查汇总用真实 seed 数、样本标准差，并验证 CIFAR-10 配对差值。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from report import summarize
from train import default_config


class ReportTests(unittest.TestCase):
    def test_aggregation_and_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1):
                for method in ("ce", "kd"):
                    cfg = default_config("cifar10") | dict(seed=seed, method=method, variant=method,
                              model="resnet56", comparison="cifar10_reversed")
                    accuracy = 0.70 + seed * 0.02 + (0.03 if method == "kd" else 0)
                    result = dict(config=cfg, test={"accuracy": accuracy}, validation={"accuracy": accuracy},
                                  data={"train_hash": "train", "val_hash": "val", "test_hash": "test"},
                                  complexity={"parameters": 100, "flops": 200}, training_seconds=1,
                                  selected_epoch=1, initial_hash=f"seed-{seed}",
                                  teacher={"sha256": "teacher"} if method == "kd" else None)
                    folder = root / f"{method}_{seed}"
                    folder.mkdir()
                    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
            summarize(root)
            def read(name):
                with (root / "summary" / name).open(encoding="utf-8-sig", newline="") as stream:
                    return list(csv.DictReader(stream))
            aggregate = read("aggregate.csv")
            self.assertEqual(len(aggregate), 2)
            for row in aggregate:
                self.assertEqual(int(row["n"]), 2)
                self.assertAlmostEqual(float(row["std"]), 0.02 / 2**0.5)
            pairs = read("paired_cifar10.csv")
            self.assertEqual(len(pairs), 2)
            for row in pairs:
                self.assertAlmostEqual(float(row["delta_pp"]), 3.0)
            self.assertAlmostEqual(float(read("paired_cifar10_summary.csv")[0]["mean_delta_pp"]), 3.0)


if __name__ == "__main__":
    unittest.main()
