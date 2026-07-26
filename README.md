# LeNet-MNIST Two-Stage Training Experiment

A small PyTorch project that trains a **LeNet-5** on **MNIST** in two stages: a *pretrain* stage, then a *finetune* stage that continues training the same model (same dataset, lower learning rate). Both stages log to TensorBoard and CSV, and are runnable end-to-end with a single CLI command.

> **Note:** the "finetune" stage currently reuses the exact same MNIST training set as pretraining, just at a lower learning rate — it's a continued-training pass, not finetuning on a different dataset or task.

---

## 📦 Project Overview

- 🔧 Framework: PyTorch
- 🧪 Dataset: MNIST, resized to 32×32 to match LeNet-5's expected input
- 📁 Model: LeNet-5 — conv → avg-pool → conv → avg-pool → fc → fc → fc, with ReLU activations (a modernized variant of the original tanh/sigmoid LeNet-5)
- 📊 Logging: TensorBoard scalars + per-epoch CSV loss logs
- 💡 Goal: exercise a two-stage (pretrain/finetune) training pipeline on a small, fast-to-train model

---

## 🗂️ Project Structure

```
lenet_mnist/
├── lenet/                # Library package
│   ├── config.py         # Default hyperparameters and paths
│   ├── data.py           # MNIST dataloaders (torchvision + preprocessing)
│   ├── model.py           # LeNet-5 architecture
│   ├── train.py           # Training loop with TensorBoard + CSV logging
│   └── evaluate.py       # Test-set accuracy
├── main.py               # CLI entry point: runs pretrain -> eval -> finetune -> eval
├── requirements.txt      # pip dependencies
├── environment.yml       # Conda environment definition
├── checkpoints/          # Saved model weights (created on first run, gitignored)
├── data/                 # MNIST dataset, auto-downloaded on first run (gitignored)
└── runs/                 # TensorBoard event files + CSV loss logs (gitignored)
```

---

## 🚀 Getting Started

### 1. Set up the environment

Using conda:

```bash
conda env create -f environment.yml
conda activate lenet-env
```

Or with pip (into an existing Python 3.9+ environment):

```bash
pip install -r requirements.txt
```

### 2. Run training and evaluation

```bash
python main.py
```

This runs, in order:
- 🧠 Pretraining on MNIST
- 📈 Evaluation on the test set
- 🔁 Continued training ("finetune") on the same training set, at `lr / 10`
- ✅ Final evaluation

MNIST is downloaded automatically into `data/` the first time you run it.

### 3. Override defaults via CLI flags

Defaults live in [`lenet/config.py`](lenet/config.py). Any of them can be overridden without editing that file:

```bash
python main.py --epochs-pretrain 10 --epochs-finetune 5 --lr 0.005 --batch-size 128
```

Available flags (`python main.py --help`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--batch-size` | 64 | DataLoader batch size |
| `--epochs-pretrain` | 5 | Epochs for the pretrain stage |
| `--epochs-finetune` | 3 | Epochs for the finetune stage |
| `--lr` | 0.01 | Pretrain learning rate (finetune uses `lr / 10`) |
| `--momentum` | 0.9 | SGD momentum |
| `--data-dir` | `data` | Where MNIST is downloaded to / read from |
| `--checkpoint-dir` | `checkpoints` | Where model weights are saved |
| `--log-dir` | `runs` | Where TensorBoard + CSV logs are written |

---

## 💾 Outputs

- **Checkpoints**: `checkpoints/lenet_pretrained.pth` and `checkpoints/lenet_finetuned.pth`
- **Logs**: `runs/pretrain/` and `runs/finetune/`, each with a TensorBoard event file and a `<phase>_log.csv` (columns: `Epoch, Loss`)

All three of `checkpoints/`, `data/`, and `runs/` are gitignored — they're regenerated locally, not checked into version control.

---

## 📊 Visualize with TensorBoard

```bash
tensorboard --logdir=runs
```

Then open [http://localhost:6006](http://localhost:6006) to see the pretrain and finetune loss curves.

---

## 🧪 Sample Results

Per-epoch training loss from a local run with the default config (batch size 64, lr 0.01, momentum 0.9, 5 pretrain + 3 finetune epochs, CPU):

| Phase | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|-------|---------|---------|---------|---------|---------|
| Pretrain | 0.4099 | 0.0808 | 0.0553 | 0.0420 | 0.0334 |
| Finetune | 0.0177 | 0.0151 | 0.0140 | — | — |

Test-set accuracy is printed to stdout after each evaluation step rather than logged to a file — run `python main.py` yourself to see the numbers for your machine.

---

## 📌 Notes

- Input images are resized to 32×32 to fit LeNet-5's expected input size.
- Training runs on GPU automatically if `torch.cuda.is_available()`, otherwise falls back to CPU — there's no Apple Silicon (MPS) path.
- Re-running `python main.py` overwrites the checkpoints and logs from the previous run (same filenames each time).

---

## 🔮 Future Ideas

- Add a validation split + early stopping
- Log test-set accuracy to CSV/TensorBoard instead of only stdout
- Try a real second dataset for the finetune stage instead of reusing the pretrain data
- Try on CIFAR-10 or CINIC-10 with a deeper model (e.g. GoogLeNet)
- Add a confusion matrix + per-class metrics

---

## 🙋 Author

This repository was built for educational and research purposes. Feel free to fork it, contribute ideas, or build your own experiments based on it!
