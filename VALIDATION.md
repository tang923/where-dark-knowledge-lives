# 实现验证记录

验证环境：本机 NVIDIA GeForce RTX 4070 Laptop GPU；Python 3.11、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。

## 已执行的检查

- `python -m unittest discover -s tests -v`：14 项通过，包括公式与梯度、数据隔离、教师验证划分检查、教师冻结、配对初始化、精确 epoch 续训、结果统计。
- `python experiments.py --track all --profile full --seeds 0 1 2 --dry-run`：生成完整正式实验清单，不启动长训练。
- `python experiments.py --track all --profile smoke --seeds 0 1 2`：45 个真实数据快速运行全部完成（MNIST 26、CIFAR-10 7、CIFAR-100 12）；每次只有 200 张训练图片、1 epoch。验证了三条主线、组件、扩展与三种子编排，准确率不用于研究结论。
- `python experiments.py --track mnist --profile pilot` 和 `--track extension --profile pilot`：共 26 个 MNIST 运行完成，每个仅 3 epochs；包含温度扫描、组件消融、缺失类、beta 扫描和无 CE 控制。
- `python report.py --root runs/pilot`：CSV 与 PNG 已生成；温度曲线和缺失类取舍图已人工查看。
- `python report.py --root runs/smoke`：确认 CIFAR-10 三个种子的初始权重与数据一致；CIFAR-100 CE/KD/DKD 各汇总三次，TCKD/NCKD 各一次，单种子标准差留空。

## 短训练示例：帮助理解输出，不作为最终实验结论

本轮 pilot 使用 5000 张训练图片、1000 张验证图片和 1000 张测试图片；单 seed=0。教师见过全部十类；缺失类学生从约 90% 的训练划分中删除数字 2 后，均衡抽取 5000 张图片。

按完整数据验证结果选定 T=2、alpha=1、beta=2。选择与缺失数字测试结果无关。

| 缺失数字 2 的实验 | 总体测试准确率 | 数字 2 召回率 | 其余九类准确率 |
|---|---:|---:|---:|
| CE | 81.2% | 0.0% | 90.22% |
| CE + KD | 87.8% | 58.0% | 91.11% |
| CE + TCKD | 82.2% | 0.0% | 91.33% |
| CE + NCKD | 90.0% | 81.0% | 91.00% |
| CE + DKD | 90.3% | 84.0% | 91.00% |
| KD，无 CE | 86.1% | 71.0% | 87.78% |
| DKD，无 CE | 88.1% | 91.0% | 87.78% |

这组例子展示了为什么需要同时看“缺失数字”和“其他九类”：去掉 CE 后，缺失数字召回率上升，但其余类别准确率下降。它只说明本次小规模运行出现了该现象；正式结论还需要 full 预算与适当重复。

原始记录在 `runs/pilot/mnist/*/result.json`，汇总在 `runs/pilot/summary/`。以上是本机短训练验证。

## MNIST 正式运行记录

Colab 下载目录 `colab/mnist-20261116T235541Z-1-001/mnist/` 包含 MNIST 基础与原扩展共 26 个 full 运行的配置、训练日志与结果。原扩展使用缺失数字 2、seed 0，在完整数据验证集上选出 T=2、β=1。

`colab/mnist要求.txt` 中的后续实验写到 `runs/local_full/`。B1 每轮从验证集中计算缺失类召回率、平均概率（T=1）、平均 logit 和平均 margin。A 的 84 次运行已在本机 RTX 4070 完成，汇总在 `mnist_omit/omitted_summary.json`。原教师的 B2 统计在 `mnist_teacher_stats/original.json`。

A 的缺失类召回率在四个数字上的平均值为 CE 0%、CE + TCKD 0%、CE + KD 84.15%、CE + NCKD 94.68%、CE + DKD 95.34%。四个数字均呈现相同的排序。数字 2、seed 0 的重跑使用同一教师和数据划分，但与原 Colab 运行的初始化哈希不同。KD 的召回率从 82.85% 到 83.43%，跨环境重跑未实现逐项精确复现。

D 的两版候选教师均在本机完成 60 轮训练。Dropout 版最高验证准确率为 98.80%（第 43 轮），加平移增强后为 98.93%（第 52 轮）。两版均低于预设的 99% 门槛，因此未评估候选教师的测试准确率，也未开始强教师的学生训练与对应的 B2 统计。

D 的弱教师对照已完成。Seed 0 复用 Colab T4 结果，seed 1 和 2 在本机 RTX 4070 上补齐，每个种子内的方法均通过配对检查。相对同种子 CE，KD 平均提高 0.10 个百分点，TCKD 提高 0.17 个百分点，NCKD 下降 0.17 个百分点，DKD 下降 0.04 个百分点。三种子汇总见 `runs/local_full/mnist_strong/paired_weak_comparison.json`。

D 的固定教师补充实验按验证集选定平移增强教师（第 52 轮，98.93%），随后对冻结的 checkpoint 做了一次测试，准确率为 99.01%（9901/10000）。记录在 `runs/local_full/mnist_strong/fixed_kd/teacher_evaluation.json`。`fixed-kd` 只运行 T=2 的 KD，seed 0、1、2 使用已有 CE 结果。Seed 0 的 CE 来自 Colab，初始化哈希与本机不同，汇总会另外报告初始化一致的 seed 1、2。

本地验证通过：14 项现有测试、84 个配置的路径与固定超参数检查、旧 full 结果的缓存复用、旧日志格式的断点续训。另用可手算的张量检验了 B1/B2 指标，并用模拟运行验证了强教师门槛、平移回退、学生参数选择和配对流程。
