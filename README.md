# ECE661 · 知识蒸馏实验

按 Project Proposal 实现：MNIST 经典 KD、CIFAR-10 反向 KD、CIFAR-100 组件消融与 DKD，以及 MNIST 缺失数字扩展。只依赖 PyTorch、torchvision、NumPy、Matplotlib；没有额外训练框架。

## 先读这几个文件

| 文件 | 负责什么 | 建议阅读顺序 |
|---|---|---|
| `losses.py` | CE、KD、TCKD、NCKD、DKD，公式的直接实现 | 1 |
| `models.py` | 两层 MLP、CIFAR ResNet、参数量/FLOPs | 2 |
| `train.py` | 所有实验共用的训练、评估、保存和续训 | 3，先看 `run()` 的 epoch 循环 |
| `data.py` | 数据增强、固定划分、删除数字 | 4 |
| `experiments.py` | proposal 中的实验组合及验证集选参 | 5 |
| `report.py` | CSV 汇总、均值/标准差、配对差值和图 | 需要写报告时看 |
| `DEFENSE.md` | 项目主线、公式解释和答辩问题 | 开始前先通读 |

## 环境与最快上手

本机已验证的解释器：`D:\programming\anaconda\envs\ml\python.exe`，PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128。
系统默认的 `python` 没有 PyTorch，因此在 PowerShell 先指定：

```powershell
$python = 'D:\programming\anaconda\envs\ml\python.exe'
& $python -m unittest discover -s tests -v
& $python experiments.py --track mnist --profile pilot
& $python report.py --root runs/pilot
```

其他机器/Colab 使用 Python 3.10+，在已经安装匹配版本 PyTorch/torchvision 的环境中执行 `python -m pip install -r requirements.txt`。Colab 上传本目录的 `.py` 文件即可，命令前加 `!`；不需要修改代码。Colab 的 `--output` 可设到已挂载的 Drive，`--data-root` 保留本地磁盘以提高读数据速度。

直接使用 Colab：参照 [COLAB.md](COLAB.md)，上传 `colab/Run_Experiments.ipynb` 和 `colab/kd_colab.zip`。Notebook 已处理 Drive 挂载、GPU 检查、依赖、训练入口及图表汇总。

## 三种运行规模

| profile | MNIST | CIFAR-10 / CIFAR-100 | 用途 |
|---|---|---|---|
| `smoke` | 200 张训练、100 张验证/测试、1 epoch | 同左 | 检查所有分支能运行；无研究结论意义 |
| `pilot` | 5000 张训练、1000 张验证/测试、3 epochs | 2000 张训练、500 张验证/测试、2 epochs | 熟悉流程、粗测运行速度 |
| `full` | 约 90% 官方训练集、20 epochs | 45000 张训练、240 epochs | 正式实验；完整独立测试集 |

各 profile 输出到不同目录。小规模子集按类别均衡抽取；删除数字后再抽取学生训练子集。pilot/smoke 结果不能与 full 混为一组，教师也不能跨 profile 复用。

## 正式实验命令

先查看清单，不下载数据、不训练：

```powershell
& $python experiments.py --track all --profile full --seeds 0 1 2 --dry-run
```

按 proposal 的两阶段顺序，先每种方法一个种子，再补主要比较，最后扩展：

```powershell
& $python experiments.py --track mnist --profile full
& $python experiments.py --track cifar10 --profile full
& $python experiments.py --track cifar100 --profile full
& $python experiments.py --track cifar10 --profile full --seeds 0 1 2
& $python experiments.py --track cifar100 --profile full --seeds 0 1 2
& $python experiments.py --track extension --profile full
& $python report.py --root runs/full
```

也可以 `--track all --profile full --seeds 0 1 2` 一次执行所有实验。额外种子只重复 CIFAR-10 的 CE/KD 和 CIFAR-100 的 CE/KD/DKD；MNIST 与其他消融使用第一个种子。教师固定 seed=0，所有学生共享它。重复原命令会跳过完成的实验，并从未完成实验的最近完整 epoch 续训。

每次训练会输出 epoch 时间。正式训练可能需要数十小时；proposal 中 30–45 GPU 小时是预算而非保证。评估时间已计入 `training_seconds`，初始教师验证、数据下载、最终测试及 checkpoint 写盘不计入，不能把它当作精确 GPU kernel 时间。可据 `history.csv` 的稳定 epoch 时间估算剩余训练预算。Windows 默认 `--workers 0`，减少启动开销并方便理解；数据加载成为瓶颈时可尝试 `--workers 2`。

## 实验到底做了什么

**MNIST。** 教师 `784→1200→1200→10`，学生 `784→800→800→10`，隐藏层用 ReLU。先分别用 CE 训练，然后用 T=1/2/4/8/20 做 KD。按完整数据验证准确率选 T；并列时选列表中较早的 T。固定随机种子 661 抽取缺失数字（当前为 2），做缺失数字 CE 和 KD。

**CIFAR-10。** 小教师 ResNet-20，大学生 ResNet-56；配对比较 CE 与 KD。CIFAR ResNet 的三个 stage 通道数为 16/32/64；不是 torchvision 面向 ImageNet 的 ResNet。

**CIFAR-100。** 教师 ResNet-56，学生 ResNet-20。跑 CE、KD、TCKD、NCKD、DKD 五组。默认从头训练教师；如果有兼容的公开 checkpoint，可用：

```powershell
& $python experiments.py --track cifar100 --profile full --cifar100-teacher path/to/resnet56.pth --teacher-source "checkpoint download URL and author" --teacher-training-data "official CIFAR-100 train split"
```

支持纯 state_dict、`model` 或 `state_dict` 包装，以及 `module.` 前缀。层名与 mdistiller 的 CIFAR ResNet 对齐，严格检查权重形状；不会静默跳过不匹配的层。只加载张量/基础类型（`weights_only=True`）。外部教师先在验证集评估并记录来源、SHA-256；它若训练过全部 50000 张图片，就可能见过我们划出的验证图片，报告必须披露。独立测试集仍不能用于调参。

**缺失数字扩展。** 先在完整 MNIST 数据上用 KD 选 T，再用 DKD 的 beta∈{1,2,4,8} 选 beta（alpha 固定 1）。保存 `temperature_selection.json` / `extension_selection.json` 后，缺失数字实验固定这些设置。主比较有 CE、KD、TCKD、NCKD、DKD；额外扫描 DKD beta，并加入 CE 权重为 0 的 KD、DKD 控制组。扩展中的主 DKD/NCKD beta 以 selection 文件为准，其他 beta 是敏感性分析，不能按缺失数字测试结果挑“最优”方法。

教师始终看到十类；学生保留十维输出，只从训练数据去掉一个数字。验证和测试仍有十类。缺失数字实验统一使用最终 epoch，避免用缺失类验证结果挑 checkpoint。输出总体准确率、缺失数字召回率、其他九类准确率和混淆矩阵，不做事后偏置校正。

## 固定的训练协议

| 设置 | MNIST | CIFAR |
|---|---|---|
| 优化器 | SGD，momentum=0.9 | SGD，momentum=0.9 |
| batch size / 初始学习率 | 128 / 0.01 | 64 / 0.05 |
| weight decay | 0 | 0.0005 |
| 降学习率 | 完成第 10、15 轮后 ×0.1 | 完成第 150、180、210 轮后 ×0.1 |
| 数据增强 | 无 | padding=4 的随机裁剪 + 水平翻转 |
| 蒸馏权重 warmup | 无 | 前 20 epochs 线性增加至 1 |
| 主实验 CE 权重 / 蒸馏权重 | 1 / 1 | 1 / 1 |

CIFAR-100 的训练日程、T=4、alpha=1、beta=2 来自 [DKD 官方模型对配置](https://github.com/megvii-research/mdistiller/blob/master/configs/cifar100/dkd/res56_res20.yaml) 和[默认配置](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/engine/cfg.py)。CIFAR-10 沿用该日程作为匹配预算的初始方案，不声称是其论文原设置。

为控制变量，所有 CIFAR 蒸馏方法使用相同 warmup，CE 权重始终为 1。这与官方经典 KD 默认 CE=0.1、KD=0.9 不同。NCKD-only 用 `beta * NCKD`，TCKD-only 用 `alpha * TCKD`，使组件消融与 DKD 中的系数一致。MLP 未加入原论文中的 dropout 等额外训练技巧。这些都是需要在报告里说明的实现选择。

每种数据集从官方训练集按类别固定留出约 10% 验证数据（split seed=661），正式测试集不参与选参。全数据实验按验证集挑 checkpoint；缺失数字实验固定最终 epoch。同 seed 的同架构学生初始权重完全相同，数据顺序和随机增强按 seed/epoch 固定。默认使用确定性运算；跨 PyTorch/CUDA 版本不保证逐位一致。

## 结果在哪里

`runs/<profile>/<dataset>/<variant>_seedN/` 内：

- `config.json`：全部实际参数及配置指纹。
- `history.csv`：每轮 loss、CE、未乘总蒸馏权重/warmup 的蒸馏项、训练/验证准确率、耗时。
- `best.pt`：最终用于测试的模型；缺失数字实验这里实际是最后一轮模型。
- `last.pt`：模型、优化器、学习率日程、随机数状态、历史记录，按完整 epoch 恢复。
- `result.json`：最终验证/测试、复杂度、初始化/数据指纹、教师来源、环境信息。

汇总目录含 `runs.csv`、`aggregate.csv`、`paired_cifar10.csv`、`paired_cifar10_summary.csv` 及 PNG 图。准确率以 0–1 保存，图以百分比显示，配对差值 `delta_pp` 为百分点。标准差使用样本标准差；单种子标准差留空并标记 exploratory，不能写成 0。

FLOPs 统一为 Conv/Linear 的 `2 × MACs`，单张图，不包括 BN/ReLU/池化/偏置加法。MLP800 参数量为 1,276,810、FLOPs 为 2,550,400；MLP1200 为 2,395,210、4,785,600。蒸馏增加训练时教师前向计算，学生推理时不需要教师。

## 修改一个实验

需要自定义预算时，可以直接用 `train.py`，或复制已有 `config.json` 修改并指定新的 `out`。例如：

```powershell
& $python train.py --dataset mnist --model mlp800 --method kd --teacher runs/full/mnist/teacher_seed0/best.pt --temperature 4 --out runs/full/mnist/manual_kd
```

同目录参数改变会报错，避免覆盖旧实验。更改 epoch 后也必须使用新目录；精确续训须保持原参数。`train.py --config path/to/config.json` 可重跑已保存配置。若要另做缩短日程的研究，用新的 output 和明确的 profile 标签，报告真实预算。

## 验证与边界

`python -m unittest discover -s tests -v` 检查逐样本 KD 分解及梯度、NCKD 对目标 logit 的零梯度、缺失类 CE 梯度、极端 logits 稳定性、教师冻结、模型复杂度、数据隔离、配对初始化和精确 epoch 续训。

当前实现是 proposal 的可复现实验工具，不预设 DKD 或反向 KD 必然获胜。smoke/pilot 只验证工程流程；正式对比应完成 full 预算与主要比较的三个种子，并如实解释负结果。此项目采用的 CIFAR 模型组合、独立验证划分和统一 CE 权重，不构成 DKD Table 1 的逐项精确复现。
