# Colab 运行步骤

本机 RTX 4070 Laptop（8 GB）已跑通本项目所有模型和当前 batch size。显存够用；完整实验的主要成本是多模型、多种子的累计训练时间。可以本地跑 MNIST、Colab 跑 CIFAR；同一组比较尽量保持同一环境。

## 第一次使用

1. 在 Google Drive 的“我的云端硬盘”里新建 `ECE661` 文件夹。
2. 上传本项目 `colab/kd_colab.zip` 到这个文件夹，保持文件名不变。无需上传本地 `data/`、`runs/` 或 PDF。
3. 打开 https://colab.research.google.com/ ，选择“文件 → 上传笔记本”，上传 `colab/Run_Experiments.ipynb`。
4. 在“运行时 → 更改运行时类型”选择可用的 GPU，例如界面提供的 T4，然后连接。不要选 TPU。
5. 按 notebook 顺序执行“准备环境”和“定义运行入口”两个代码单元。挂载 Drive 时按界面授权。
6. 第一次保持 `TRACK="mnist"`、`PROFILE="pilot"`、`SEEDS=[0]`，执行训练单元，再执行汇总单元。
7. 查看 `runs.csv` 和温度图，理解学生基线与 KD 的区别后，再将 `PROFILE` 改为 `"full"`。

代码解压到 `/content/ece661`，数据下载到 `/content/ece661_data`，checkpoint 和结果写到 `/content/drive/MyDrive/ECE661/runs_colab`。数据不需要逐张从 Drive 读取；每轮完成后将 checkpoint 写入 Drive。检查 Drive 可用空间，保留 checkpoint 文件。

## 正式实验顺序

每次只修改 notebook 训练单元的这三个变量，再运行该单元：

| 次序 | TRACK | PROFILE | SEEDS | 目的 |
|---|---|---|---|---|
| 1 | `mnist` | `full` | `[0]` | 教师、学生、五个温度、基础缺失数字实验 |
| 2 | `cifar10` | `full` | `[0]` | 小教师、大学生 CE/KD |
| 3 | `cifar100` | `full` | `[0]` | 教师、CE/KD/TCKD/NCKD/DKD |
| 4 | `cifar10` | `full` | `[0, 1, 2]` | 复用 seed 0，补主要比较 |
| 5 | `cifar100` | `full` | `[0, 1, 2]` | 复用 seed 0，补 CE/KD/DKD |
| 6 | `extension` | `full` | `[0]` | 复用 MNIST 教师与温度实验，做组件与无 CE 控制 |

如果 MNIST 留在本地，就把第 1、6 步都留在本地；Colab 从第 2 步开始。不要直接搬动未完成运行的目录去续训：checkpoint 记录了配置与教师路径，跨平台路径不同。两台机器各跑完整 track，最后汇总结果即可。

notebook 默认只跑 MNIST pilot；“运行全部”不会自动启动所有正式训练。完整 CIFAR 日程为每个模型 240 epochs，MNIST 为 20 epochs。观察正式运行前几轮的耗时后，再估算单模型训练时间；pilot 数据量较小，不能直接拿其 epoch 时间乘正式 epoch 数。

## 断线后怎么继续

重新连接 GPU，重跑前两个环境/入口单元，然后把 `TRACK`、`PROFILE`、`SEEDS` 改回中断前的值，运行训练单元。

同一个输出目录、同样参数下，已完成实验显示 `[cached]`；未完成实验显示 `[resume] ... epoch N`，从最近保存的完整 epoch 继续。当前未完成的 epoch 会重跑。数据缓存可能需要重新下载，已保存的 Drive checkpoint 不需要重新训练。

在同一批实验中固定代码、依赖版本、种子顺序、batch size 和 worker 数；不要在两个会话中同时写同一个运行目录。若改变训练方案，使用新的 `RUN_ROOT`，保留原实验。

## 结果与检查

- Drive：`ECE661/runs_colab/full/<dataset>/<experiment>/result.json`。
- 图表和 CSV：`ECE661/runs_colab/full/summary/`。
- 每次环境启动的 Python/PyTorch/torchvision/GPU 与代码包校验信息：`ECE661/runs_colab/colab_environment_*.json`。
- notebook 的“训练进度”单元可以查看已完成多少轮，并粗估已启动模型的剩余时间；不包含尚未启动的模型或后续种子。
- 完整实验结果用于最终报告；`pilot`、`smoke` 只用于检查流程。

当前 notebook 已在本地检查 JSON 格式、代码语法、命令参数和代码包完整性；Google Drive 挂载与 Colab GPU 分配需在你的账号中执行，尚未代你在 Colab 上运行。

## 本地也可以继续跑

```powershell
& 'D:\programming\anaconda\envs\ml\python.exe' experiments.py --track mnist --profile full
& 'D:\programming\anaconda\envs\ml\python.exe' experiments.py --track extension --profile full
& 'D:\programming\anaconda\envs\ml\python.exe' report.py --root runs/full
```

Colab 的 GPU 类型和使用额度会动态变化，免费会话可能提前结束；官方给出的免费运行上限为最多 12 小时，并非保证时长。具体速度应以当前分配的 GPU 和实际 epoch 日志为准。[Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html)

改过代码后，在项目根目录执行 `python colab/build_bundle.py` 重新打包，再上传新 ZIP；进行中的正式实验应继续使用原代码包。
