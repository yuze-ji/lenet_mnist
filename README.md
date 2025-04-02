# LeNet-MNIST Two-Stage Training Experiment

This project implements a **two-stage training pipeline** (pretraining + finetuning) using the classic **LeNet-5** architecture on the **MNIST** dataset. It demonstrates how a model can benefit from representation learning in the pretraining stage, followed by performance refinement in the finetuning stage.

---

## 📦 Project Overview

- 🔧 Framework: PyTorch
- 📊 Visualizations: TensorBoard
- 🧪 Dataset: MNIST (resized to 32×32 for LeNet compatibility)
- 📁 Model: LeNet-5
- 💡 Goal: Validate two-stage training effectiveness

---

## 🗂️ Folder Structure

```
lenet_mnist_experiment_v3_logs/
├── configs.py               # Configurations: batch size, epochs, LR, etc.
├── data_utils.py            # Data loader with torchvision + preprocessing
├── environment.yml          # Conda environment definition
├── evaluate.py              # Accuracy calculation on test set
├── lenet_pretrained.pth     # Saved pretrained model
├── main.py                  # Entry point for training + finetuning + eval
├── model.py                 # LeNet-5 model architecture
├── requirements.txt         # pip dependencies (alternative to conda)
├── train.py                 # Training loop with TensorBoard and CSV logging
└── runs/                    # TensorBoard logs and CSV loss tracking
```

---

## 🚀 Getting Started

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate lenet-env
```

### 2. Run Training and Evaluation

```bash
python main.py
```

This performs:
- 🧠 Pretraining on MNIST
- 📈 Evaluation on test set
- 🔁 Finetuning on the same dataset
- ✅ Final evaluation with updated weights

---

## 📊 Visualize with TensorBoard

To view the training and finetuning loss curves:

```bash
tensorboard --logdir=runs
```

Then open your browser at [http://localhost:6006](http://localhost:6006)

---

## 🧪 Sample Results

| Phase      | Final Loss | Accuracy   |
|------------|-------------|------------|
| Pretrain   | ~0.0386     | ~98.75%    |
| Finetune   | ~0.0112     | ~99.02% ✅ |

> Finetuning phase further reduces the loss and improves accuracy, validating the two-stage approach.

---

## 📌 Notes

- Input images are resized to 32×32 to fit LeNet's expected input.
- CSV logs are stored at:
  - `runs/pretrain/pretrain_log.csv`
  - `runs/finetune/finetune_log.csv`
- Model is saved as `lenet_pretrained.pth` after pretraining.

---

## 🔮 Future Ideas

- Add validation split + early stopping
- Try on CIFAR-10 or CINIC-10 with GoogLeNet
- Use transfer learning across datasets (MNIST → custom digits)
- Add confusion matrix + detailed metrics

---

## 🙋‍♀️ Author

This repository was built for educational and research purposes.  
Feel free to fork it, contribute ideas, or build your own experiments based on it!

---