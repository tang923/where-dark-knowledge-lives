import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab"


def cell(kind, content):
    result = dict(cell_type=kind, metadata={}, source=textwrap.dedent(content).strip().splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    cells = [
        cell("markdown", """
        # ECE661：知识蒸馏实验

        **第一次使用：** 将 `kd_colab.zip` 上传至 Google Drive 的 `我的云端硬盘/ECE661/`，然后选择 **运行时 → 更改运行时类型 → GPU**。

        依次执行下面的单元。默认只运行 MNIST 的 3-epoch pilot；不会自动启动所有长训练。
        完整说明见代码包中的 `COLAB.md`，原理与答辩准备见 `DEFENSE.md`。
        """),
        cell("markdown", """
        ## 1. 准备环境（每次新会话重跑）

        挂载 Drive、解压代码、确认 GPU。原始图片放在 Colab 本地磁盘；模型与结果保存在 Drive。
        """),
        cell("code", '''
        from google.colab import drive
        from pathlib import Path
        from datetime import datetime, timezone
        import hashlib
        import importlib.metadata
        import json
        import os
        import subprocess
        import sys
        import zipfile

        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/ECE661")
        ARCHIVE = DRIVE_ROOT / "kd_colab.zip"
        CODE_ROOT = Path("/content/ece661")
        DATA_ROOT = Path("/content/ece661_data")
        RUN_ROOT = DRIVE_ROOT / "runs_colab"
        assert ARCHIVE.is_file(), f"请先把 kd_colab.zip 上传到 {DRIVE_ROOT}"
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE) as bundle:
            for name in bundle.namelist():
                target = (CODE_ROOT / name).resolve()
                if not target.is_relative_to(CODE_ROOT.resolve()):
                    raise ValueError(f"Invalid archive member: {name}")
            bundle.extractall(CODE_ROOT)
        os.chdir(CODE_ROOT)
        manifest = json.loads((CODE_ROOT / "bundle_manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            actual = hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest()
            assert actual == expected, f"代码包校验失败: {name}"
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
        import torch
        assert torch.cuda.is_available(), "当前未连接 GPU，请更改运行时类型后重新执行此单元。"
        environment = {
            "python": sys.version,
            "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "matplotlib"]},
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "cuda": torch.version.cuda,
            "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (RUN_ROOT / f"colab_environment_{stamp}.json").write_text(json.dumps(environment, indent=2))
        print(json.dumps(environment, indent=2))
        print("Checkpoint / results:", RUN_ROOT)
        '''),
        cell("markdown", "## 2. 定义运行入口，并检查核心代码（每次新会话重跑）"),
        cell("code", '''
        WORKERS = 2

        def run_track(track, profile="full", seeds=(0,), dry_run=False):
            command = [sys.executable, "-u", "experiments.py", "--track", track,
                       "--profile", profile, "--seeds", *map(str, seeds),
                       "--output", str(RUN_ROOT), "--data-root", str(DATA_ROOT),
                       "--device", "cuda", "--workers", str(WORKERS)]
            if dry_run:
                command.append("--dry-run")
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=CODE_ROOT, check=True)

        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=CODE_ROOT, check=True)
        '''),
        cell("markdown", """
        ## 3. 选择这次要做的实验

        第一次保持下面的默认值。pilot 验证流程后再切换 full；每次只跑一条主线。

        | 次序 | TRACK | PROFILE | SEEDS |
        |---|---|---|---|
        | 先检查 | mnist | pilot | [0] |
        | 1 | mnist | full | [0] |
        | 2 | cifar10 | full | [0] |
        | 3 | cifar100 | full | [0] |
        | 4 | cifar10 | full | [0, 1, 2] |
        | 5 | cifar100 | full | [0, 1, 2] |
        | 6 | extension | full | [0] |

        CIFAR 完整日程为每个模型 240 epochs。增加种子时保留 seed 0，它会自动复用。
        如果 MNIST 在本地运行，extension 也放本地，Colab 从 cifar10 开始。
        """),
        cell("code", '''
        TRACK = "mnist"
        PROFILE = "pilot"
        SEEDS = [0]

        run_track(TRACK, PROFILE, SEEDS)
        '''),
        cell("markdown", """
        ## 4. 汇总当前 profile 的结果

        CSV、PNG 保存在 Drive。数据列中的准确率为 0–1；图表使用百分比。
        只汇总已完成的实验，部分运行可继续训练后再重新汇总。
        """),
        cell("code", '''
        import csv
        from IPython.display import Image, display

        result_root = RUN_ROOT / PROFILE
        if not list(result_root.glob("*/*/result.json")):
            print("还没有完成的实验；稍后再执行此单元。")
        else:
            subprocess.run([sys.executable, "report.py", "--root", str(result_root)],
                           cwd=CODE_ROOT, check=True)
            summary = result_root / "summary"
            with (summary / "aggregate.csv").open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    sd = f" ± {float(row['std'])*100:.2f}%" if row["std"] else " (single seed)"
                    print(f"{row['dataset']:8} {row['variant']:26} n={row['n']}  {float(row['mean'])*100:.2f}%{sd}")
            for pattern in ["*_temperature.png", "*_tradeoff.png", "*cifar100_components*.png"]:
                for path in sorted(summary.glob(pattern)):
                    display(Image(filename=str(path)))
            print("所有图表和 CSV:", summary)
        '''),
        cell("markdown", """
        ## 5. 查看训练进度（可选）

        单模型粗估仅使用该运行最近三轮的日志，不包含尚未启动的模型或其他种子。
        正式运行结束前，停止训练单元后也可执行本单元；不要同时启动两个训练进程写同一目录。
        """),
        cell("code", '''
        import csv
        import statistics

        for path in sorted((RUN_ROOT / PROFILE).glob("*/*/history.csv")):
            with path.open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            if not history:
                continue
            cfg = json.loads((path.parent / "config.json").read_text())["config"]
            done = int(history[-1]["epoch"])
            seconds = statistics.mean(float(row["seconds"]) for row in history[-3:])
            remaining = max(0, cfg["epochs"] - done) * seconds / 3600
            state = "完成" if (path.parent / "result.json").exists() else "进行中/待续训"
            print(f"{path.parent.parent.name}/{path.parent.name}: {done}/{cfg['epochs']} epochs, "
                  f"{seconds:.1f}s/epoch, 剩余约 {remaining:.2f}h, {state}")
        '''),
        cell("markdown", """
        ## 断线后的操作

        1. 重新连接 GPU，执行前两个环境/入口单元。
        2. 将训练单元的 TRACK、PROFILE、SEEDS 恢复到中断前的值。
        3. 重跑训练单元；看到 `[resume]` 表示从最近完整 epoch 续训，`[cached]` 表示跳过已完成实验。

        保持 RUN_ROOT、代码与参数不变，checkpoint 已写入 Drive。当前未完成的 epoch 会重跑；临时磁盘的数据可能重新下载。
        同一批实验尽量保持软件版本和 GPU 环境一致；跨环境不保证逐位一致。

        Colab GPU 和配额不保证，免费会话最多 12 小时且可能提前结束。见 [官方 FAQ](https://research.google.com/colaboratory/faq.html)。
        """),
    ]
    for i, item in enumerate(cells):
        item["id"] = f"ece661-{i:02d}"
    notebook = dict(cells=cells, metadata={
        "accelerator": "GPU", "colab": {"name": "Run_Experiments.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}}, nbformat=4, nbformat_minor=5)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Run_Experiments.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {OUT / 'Run_Experiments.ipynb'}")


def build_bundle():
    names = ["data.py", "models.py", "losses.py", "train.py", "experiments.py", "report.py",
             "mnist_followup.py",
             "requirements.txt", "README.md", "VALIDATION.md", "THIRD_PARTY.md", "COLAB.md",
             "tests/test_core.py", "tests/test_report.py"]
    manifest = {"sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    with zipfile.ZipFile(OUT / "kd_colab.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.write(ROOT / name, name)
        bundle.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print(f"Created {OUT / 'kd_colab.zip'} ({(OUT / 'kd_colab.zip').stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Colab notebook and code bundle.")
    parser.add_argument("--bundle-only", action="store_true", help="Keep the existing notebook")
    args = parser.parse_args()
    if not args.bundle_only:
        build_notebook()
    build_bundle()
