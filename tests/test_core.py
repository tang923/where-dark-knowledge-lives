"""检验公式、数据隔离、指标和断点续训；不下载数据、不占 GPU。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from data import balanced_limit, split_indices
from losses import kd_parts, objective
from models import complexity, make_model
from train import default_config, load_teacher, metrics_from_confusion, run, validate_config


torch.set_num_threads(2)


class LossTests(unittest.TestCase):
    def test_exact_per_sample_decomposition_and_gradient(self):
        torch.manual_seed(3)
        target = torch.arange(7) % 5
        for temperature in (1, 2, 4, 8, 20):
            student = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
            teacher = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
            parts = kd_parts(student, teacher, target, temperature)
            decomposed = parts["tckd"] + parts["other_probability"] * parts["nckd"]
            torch.testing.assert_close(parts["kd"], decomposed, atol=1e-12, rtol=1e-10)
            grad_kd = torch.autograd.grad(parts["kd"].sum(), student, retain_graph=True)[0]
            grad_parts = torch.autograd.grad(decomposed.sum(), student)[0]
            torch.testing.assert_close(grad_kd, grad_parts, atol=1e-12, rtol=1e-10)
            self.assertIsNone(teacher.grad)

    def test_nckd_excludes_target_probability(self):
        student = torch.randn(4, 10, requires_grad=True)
        target = torch.tensor([0, 2, 4, 7])
        parts = kd_parts(student, torch.randn(4, 10), target)
        grad = torch.autograd.grad(parts["nckd"].sum(), student)[0]
        torch.testing.assert_close(grad.gather(1, target[:, None]), torch.zeros(4, 1))

    def test_missing_class_ce_gradient(self):
        logits = torch.randn(5, 10, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0, 1, 2, 3, 4])
        loss, _, _ = objective(logits, target)
        grad = torch.autograd.grad(loss, logits)[0]
        torch.testing.assert_close(grad[:, 9], logits.softmax(1)[:, 9] / 5)
        self.assertTrue((grad[:, 9] > 0).all())

    def test_finite_extreme_logits_and_frozen_teacher(self):
        student = torch.tensor([[10000., -10000., 0.], [-10000., 10000., 0.]], requires_grad=True)
        teacher = (-student.detach()).requires_grad_()
        for method in ("kd", "tckd", "nckd", "dkd"):
            student.grad = None
            loss, _, _ = objective(student, torch.tensor([0, 1]), teacher, method=method)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(student.grad).all())
            self.assertIsNone(teacher.grad)

    def test_equal_predictions_zero_distillation(self):
        logits = torch.randn(8, 100)
        target = torch.arange(8)
        for method in ("kd", "tckd", "nckd", "dkd"):
            _, _, distill = objective(logits, target, logits, method=method)
            self.assertAlmostEqual(distill.item(), 0.0, places=6)


class DataModelTests(unittest.TestCase):
    def test_stratified_disjoint_fixed_split_and_omission(self):
        target = torch.arange(1000) % 10
        train, val = split_indices(target)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(1000)))
        self.assertEqual((train, val), split_indices(target))
        self.assertEqual(torch.bincount(target[val]).tolist(), [10] * 10)
        omitted = [i for i in train if int(target[i]) != 6]
        limited = balanced_limit(omitted, target, 90)
        self.assertNotIn(6, target[limited].tolist())
        self.assertEqual(len(limited), 90)
        self.assertEqual(len(set(target[limited].tolist())), 9)

    def test_mlp_complexity_is_analytic(self):
        for width in (800, 1200):
            model = make_model(f"mlp{width}", 10)
            stats = complexity(model, (1, 28, 28))
            self.assertEqual(stats["parameters"], width**2 + 796 * width + 10)
            self.assertEqual(stats["flops"], 2 * (width**2 + 794 * width))
            self.assertTrue(model.training)

    def test_cifar_shapes_and_sizes(self):
        counts = []
        for name in ("resnet20", "resnet56"):
            model = make_model(name, 100).eval()
            self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 32)).shape), (2, 100))
            counts.append(complexity(model, (3, 32, 32))["parameters"])
        self.assertGreater(counts[1], counts[0])

    def test_omitted_metrics(self):
        cm = torch.tensor([[8, 2, 0], [1, 4, 5], [0, 3, 7]])
        metrics = metrics_from_confusion(cm, 1)
        self.assertAlmostEqual(metrics["accuracy"], 19 / 30)
        self.assertAlmostEqual(metrics["omitted_recall"], 0.4)
        self.assertAlmostEqual(metrics["other_accuracy"], 0.75)

    def test_protocol_guards(self):
        with self.assertRaises(ValueError):
            validate_config(default_config() | dict(omit=2, selection="best"))
        with self.assertRaises(ValueError):
            validate_config(default_config() | dict(method="nckd", ce_weight=0, teacher="dummy"))
        with self.assertRaises(ValueError):
            validate_config(default_config() | dict(train_limit=100))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generator = torch.Generator().manual_seed(111)
        dataset = TensorDataset(torch.randn(20, 1, 28, 28, generator=generator), torch.arange(20) % 10)
        self.data_info = {"train_hash": "test-train", "val_hash": "test-val", "test_hash": "test-test"}
        self.patch = patch("train.make_datasets", return_value=((dataset, dataset, dataset), self.data_info))
        self.patch.start()
        self.cfg = default_config() | dict(epochs=2, batch_size=10, milestones=[1],
                    profile="smoke", device="cpu", seed=7, lr=0.01, download=False)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_exact_epoch_resume_and_cache_guard(self):
        continuous = self.cfg | dict(out=str(self.root / "continuous"))
        resumed = self.cfg | dict(out=str(self.root / "resumed"))
        expected = run(continuous)
        self.assertIsNone(run(resumed, stop_after=1))
        actual = run(resumed)
        self.assertEqual(expected["test"], actual["test"])
        self.assertEqual(expected["initial_hash"], actual["initial_hash"])
        for left, right in zip(expected["history"], actual["history"]):
            self.assertEqual(left["loss"], right["loss"])
        a = torch.load(self.root / "continuous/last.pt", weights_only=True)["model"]
        b = torch.load(self.root / "resumed/last.pt", weights_only=True)["model"]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(run(resumed), actual)
        with self.assertRaises(ValueError):
            run(resumed | dict(lr=0.2))

    def test_paired_student_and_frozen_teacher(self):
        teacher_path = self.root / "teacher.pt"
        teacher_cfg = self.cfg | dict(model="mlp1200", role="teacher")
        torch.save(dict(model=make_model("mlp1200", 10).state_dict(), config=teacher_cfg), teacher_path)
        cfg = self.cfg | dict(epochs=1, teacher=str(teacher_path), method="dkd")
        teacher, _ = load_teacher(cfg, 10, torch.device("cpu"))
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        baseline = run(self.cfg | dict(epochs=1, out=str(self.root / "baseline")))
        distilled = run(cfg | dict(out=str(self.root / "distilled")))
        self.assertEqual(baseline["initial_hash"], distilled["initial_hash"])
        self.assertIsNotNone(distilled["teacher"]["validation"])

    def test_teacher_validation_split_guard(self):
        teacher_path = self.root / "wrong_split.pt"
        teacher_cfg = self.cfg | dict(model="mlp1200", split_seed=123)
        torch.save(dict(model=make_model("mlp1200", 10).state_dict(), config=teacher_cfg), teacher_path)
        with self.assertRaisesRegex(ValueError, "validation split"):
            load_teacher(self.cfg | dict(teacher=str(teacher_path)), 10, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
