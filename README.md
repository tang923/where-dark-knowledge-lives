# Where Does Dark Knowledge Live?

**A controlled study of classical, reversed, and decoupled knowledge distillation**

Yuyao Tang, Xianghao Zheng, Ziyang Qu · Duke University · ECE 661 (Fall 2026) course project

**[Report (PDF)](report.pdf)** · **[Supplementary material (PDF)](supplementary.pdf)**

Knowledge distillation (KD) is usually explained by the "dark knowledge" in a
teacher's probabilities over incorrect classes. This project asks which part of
the teacher's output a student actually uses. We split the KD loss into a
target-class term (TCKD) and a non-target-class term (NCKD), derive their
gradients on individual logits, and test the predictions under one matched
protocol: within each comparison, students share initial weights, data order,
augmentation, optimizer, budget, and cross-entropy weight.

| Setting | Teacher → Student | Question |
|---|---|---|
| MNIST | MLP-1200 → MLP-800 | Does KD help when the teacher is (or is not) more accurate? How does temperature matter? |
| CIFAR-10 | ResNet-20 → ResNet-56 | Can a *weaker* teacher help a stronger student (reversed KD)? |
| CIFAR-100 | ResNet-56 → ResNet-20 | Which component (TCKD, NCKD, DKD) carries the gain? |
| MNIST, one digit removed | MLP-1200 → MLP-800 | Which component transfers knowledge of a class the student never sees? |

## Key findings

- **TCKD cannot convey which wrong class the teacher favors.** Its gradient on
  every non-target logit is proportional to the student's own probability;
  KD passes the teacher's ranking of wrong classes on only in proportion to the
  teacher's non-target mass `1 - p_y`.
- **A teacher without an accuracy advantage adds little.** Reversed KD on
  CIFAR-10 changes accuracy by `+0.12 ± 0.09` points. On MNIST, the same student
  gains `+0.10` points from a teacher as accurate as itself (98.19%) and
  `+0.48` points from a stronger one (99.01%).
- **Non-target knowledge carries the CIFAR-100 gain.** NCKD alone recovers 68%
  of KD's `+2.70`-point gain, while TCKD alone lowers accuracy. DKD matches but
  does not beat KD when both use the same cross-entropy weight; restoring the
  official KD weights reproduces the published KD baseline.
- **An unseen class is learned only through non-target knowledge.** With one
  digit removed from the student's data, recall on that digit is 0% for CE and
  TCKD, 84.2% for KD, and 95.3% for DKD, with no exception across four digits
  and three seeds. The MNIST teacher's mean non-target mass is only 0.04 (vs.
  0.69 on CIFAR-100), which explains why decoupling matters here but not for
  CIFAR-100 accuracy.

### Main results

CIFAR-100, ResNet-56 teacher (72.40%) → ResNet-20, test accuracy (%) at the final
epoch, mean ± SD over three seeds:

| CE | TCKD | NCKD | KD | KD (official weights) | DKD |
|---|---|---|---|---|---|
| 68.41 ± 0.16 | 67.57 ± 0.07 | 70.24 ± 0.34 | **71.11 ± 0.04** | 70.62 ± 0.19 | 70.95 ± 0.38 |

MNIST with one digit removed from the student's training data, test recall (%)
on the omitted digit, mean ± SD over digits {2, 3, 7, 9} × three seeds:

| CE | CE + TCKD | CE + KD | CE + NCKD | CE + DKD | DKD, no CE |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 84.2 ± 1.7 | 94.7 ± 0.3 | 95.3 ± 0.3 | **96.0 ± 0.4** |

## Repository layout

| Path | Contents |
|---|---|
| `losses.py` | CE, KD, TCKD, NCKD, and DKD, computed in the log domain |
| `models.py` | MLPs (incl. `mlp1200d` with dropout), CIFAR ResNets, parameter and FLOP counts |
| `data.py` | Fixed stratified train/validation split, augmentation, omitted-digit filtering |
| `train.py` | Shared training loop: matched initialization, exact resumption, evaluation, logging |
| `experiments.py` | The planned experiment grid (MNIST, CIFAR-10, CIFAR-100, omitted-digit extension) |
| `mnist_followup.py` | MNIST follow-ups: four omitted digits × three seeds, teacher statistics, strong teacher |
| `report.py` | CSV summaries, mean ± SD, paired differences, and plots |
| `tests/` | Unit tests for the loss decomposition, gradients, data isolation, and resumption |
| `results/full/` | CIFAR-10 and CIFAR-100 results, final-epoch evaluations, teacher statistics |
| `results/full/cifar_followup/scripts/` | Scripts for the CIFAR follow-up experiments |
| `colab/mnist-20261116T235541Z-1-001/mnist/` | Original MNIST runs (seed 0): temperature and β sweeps |
| `runs/local_full/` | MNIST follow-up results (omitted digits, extra seeds, strong teacher, teacher statistics) |
| `runs/smoke/`, `runs/pilot/` | Short pipeline checks only; not used for any conclusion |
| `colab/Run_Experiments.ipynb` | Notebook for running the pipeline on Google Colab |
| `THIRD_PARTY.md` | Sources and attribution |

Each run folder contains `config.json` (all settings and a configuration
fingerprint), `history.csv` (per-epoch losses, accuracies, and timing), and
`result.json` (validation and test metrics, confusion matrices, complexity,
teacher provenance, and environment). Model checkpoints are not included; rerun
the commands below to regenerate them.

## Setup

Python 3.10+ with PyTorch and torchvision installed for your CUDA version, then:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Reproducing the experiments

**Main grid.** Preview the plan without downloading data or training, then run each track:

```bash
python experiments.py --track all --profile full --seeds 0 1 2 --dry-run
python experiments.py --track mnist --profile full
python experiments.py --track extension --profile full
python experiments.py --track cifar10 --profile full --seeds 0 1 2
python report.py --root runs/full
```

Rerunning a command skips finished runs and resumes unfinished ones from the
last completed epoch. Use `--profile smoke` or `--profile pilot` for quick
checks on small subsets.

**CIFAR-100 teacher.** We use the public ResNet-56 checkpoint released with
[Decoupled Knowledge Distillation](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints)
(`cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth`):

```bash
python experiments.py --track cifar100 --profile full --seeds 0 1 2 \
  --cifar100-teacher path/to/resnet56.pth \
  --teacher-source "https://github.com/megvii-research/mdistiller/releases/tag/checkpoints" \
  --teacher-training-data "official CIFAR-100 training set"
```

The loader accepts a plain `state_dict` or one wrapped under `model` or
`state_dict`, and loads tensors only (`weights_only=True`). If the original file
fails to load because of pickled metadata, re-save its `model` state dict first.

**MNIST follow-ups.** `--source` points to the folder holding the original
seed-0 MNIST runs, including `teacher_seed0/best.pt`:

```bash
python mnist_followup.py omitted            --root runs/local_full --source runs/full/mnist
python mnist_followup.py omitted-summary    --root runs/local_full --source runs/full/mnist
python mnist_followup.py teacher-stats      --root runs/local_full --source runs/full/mnist
python mnist_followup.py weak-students      --root runs/local_full --source runs/full/mnist
python mnist_followup.py strong-teacher     --root runs/local_full --source runs/full/mnist
python mnist_followup.py fixed-teacher-test --root runs/local_full --source runs/full/mnist
python mnist_followup.py fixed-kd           --root runs/local_full --source runs/full/mnist
```

**CIFAR follow-ups.** The scripts in `results/full/cifar_followup/scripts/`
import the project modules, so run them with the repository root on the path:

```bash
PYTHONPATH=. python results/full/cifar_followup/scripts/eval_last.py --root runs/full/cifar100
PYTHONPATH=. python results/full/cifar_followup/scripts/teacher_confidence.py \
  --project . --config runs/full/cifar100/kd_seed0/config.json --output results/teacher_confidence
python results/full/cifar_followup/scripts/run_ef.py --project . --plan-only
```

On Windows PowerShell, set `$env:PYTHONPATH="."` instead of the `PYTHONPATH=.` prefix.

## Protocol

- **Splits.** A class-stratified 10% of each official training set is held out
  for validation (split seed 661). Test sets are used only for final evaluation.
- **Training.** MNIST: SGD, batch 128, lr 0.01, 20 epochs. CIFAR: SGD, batch 64,
  lr 0.05, weight decay 5e-4, 240 epochs with decays at 150/180/210, following
  the official DKD configuration. All methods use a cross-entropy weight of 1
  unless stated otherwise.
- **Selection.** CIFAR and omitted-digit results use the final epoch; full-data
  MNIST runs use the best validation checkpoint. Temperature (T = 2) and
  β (= 1) for MNIST were chosen on full-data validation accuracy only.
- **Pairing.** Within a seed, all students of an architecture start from
  identical weights (checked by hash) and see the same data order.

## References

- G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
- L. Yuan et al. *Revisiting Knowledge Distillation via Label Smoothing Regularization.* CVPR 2020.
- B. Zhao et al. *Decoupled Knowledge Distillation.* CVPR 2022. Code: [mdistiller](https://github.com/megvii-research/mdistiller).

## License

Released under the [MIT License](LICENSE). Third-party sources and checkpoints are attributed in [THIRD_PARTY.md](THIRD_PARTY.md).
