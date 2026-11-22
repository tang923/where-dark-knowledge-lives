# Running Experiments in Colab

All models in this project were tested locally on an 8 GB RTX 4070 Laptop GPU at the configured batch sizes. The main cost of a full experiment is the time it takes to train several models across multiple seeds. You can run MNIST locally and CIFAR in Colab. Use the same environment for all runs within a comparison when possible.

## First-time setup

1. Create an `ECE661` folder in **My Drive**.
2. Upload `colab/kd_colab.zip` to that folder and keep the filename. You can leave the local `data/` and `runs/` folders and PDF files on your machine.
3. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload `colab/Run_Experiments.ipynb`.
4. Under **Runtime → Change runtime type**, select an available GPU such as a T4, then connect. This notebook requires a GPU runtime.
5. Run the first two code cells in order. They set up the environment and define the training entry point. Grant access when prompted to mount Google Drive.
6. For your first run, keep `TRACK="mnist"`, `PROFILE="pilot"`, and `SEEDS=[0]`. Run the training cell, then the summary cell.
7. Review `runs.csv` and the temperature plot to compare the student baseline with KD, then change `PROFILE` to `"full"`.

The notebook extracts the code to `/content/ece661`, downloads the data to `/content/ece661_data`, and saves checkpoints and results to `/content/drive/MyDrive/ECE661/runs_colab`. Training reads data from the runtime's local disk and writes a checkpoint to Drive after each epoch. Check the available Drive space and keep the checkpoints for resuming interrupted runs.

## Full experiment order

For each step, set these three variables in the notebook's training cell and run it.

| Step | TRACK | PROFILE | SEEDS | What runs |
|---|---|---|---|---|
| 1 | `mnist` | `full` | `[0]` | Teacher and student runs, five temperatures, and the basic omitted-digit experiment |
| 2 | `cifar10` | `full` | `[0]` | Smaller teacher and larger student, with CE and KD |
| 3 | `cifar100` | `full` | `[0]` | Teacher and five student runs with CE, KD, TCKD, NCKD, and DKD |
| 4 | `cifar10` | `full` | `[0, 1, 2]` | Reuse seed 0 and add two seeds to the main comparison |
| 5 | `cifar100` | `full` | `[0, 1, 2]` | Reuse seed 0 and add two seeds for CE, KD, and DKD |
| 6 | `extension` | `full` | `[0]` | Reuse the MNIST teacher and temperature runs for the component and no-CE controls |

If you run MNIST locally, keep steps 1 and 6 on the local machine and start with step 2 in Colab. Checkpoints record the run configuration and teacher paths, so keep each unfinished run in its original environment. Complete each track on one machine, then combine the results.

With the notebook's default settings, **Run all** trains only the MNIST pilot. The full schedule uses 240 epochs per CIFAR model and 20 per MNIST model. Estimate training time from the first few epochs of a full run because pilot runs use less data.

## Resume after a disconnect

Reconnect to a GPU, rerun the first two setup cells, and restore the previous values of `TRACK`, `PROFILE`, and `SEEDS`. Then run the training cell.

With the same output directory and settings, completed runs show `[cached]`. Interrupted runs show `[resume] ... epoch N` and continue from the last saved epoch. The interrupted epoch runs again. The runtime may need to download the dataset again, but it can reuse the checkpoints in Drive.

Keep the code, dependency versions, seed order, batch size, and number of workers fixed within an experiment. Use only one session per run directory. If you change the training setup, use a new `RUN_ROOT` so the earlier results stay available.

## Results and progress

- Per-run results in Drive are saved to `ECE661/runs_colab/full/<dataset>/<experiment>/result.json`.
- Plots and CSV files are saved to `ECE661/runs_colab/full/summary/`.
- The Python, PyTorch, torchvision, and GPU details, along with the code archive checksum, are saved at startup to `ECE661/runs_colab/colab_environment_*.json`.
- The notebook's training progress cell shows completed epochs and estimates the remaining time for runs that have started. The estimate excludes runs and seeds that have yet to start.
- Use full runs for the final report. The `pilot` and `smoke` profiles are for checking the workflow.

The notebook JSON, Python syntax, command-line arguments, and code archive integrity were checked locally. Mounting Drive and assigning a GPU take place when you run the notebook in your Colab account.

## Run locally

```powershell
& 'D:\programming\anaconda\envs\ml\python.exe' experiments.py --track mnist --profile full
& 'D:\programming\anaconda\envs\ml\python.exe' experiments.py --track extension --profile full
& 'D:\programming\anaconda\envs\ml\python.exe' report.py --root runs/full
```

Colab GPU availability and usage limits vary. Free sessions can end early and are limited to at most 12 hours. Estimate training speed from the assigned GPU and the epoch logs. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

After changing the code, run `python colab/build_bundle.py` from the repository root and upload the new ZIP. Keep using the original code archive for any full experiment already in progress.
