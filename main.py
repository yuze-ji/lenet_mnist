# main.py

from model import LeNet5
from data_utils import get_mnist_loaders
from train import train
from evaluate import evaluate
import configs
import torch
import os

def main():
    train_loader, test_loader = get_mnist_loaders(configs.batch_size)

    print("==> Pretraining LeNet5 on MNIST...")
    model = LeNet5()
    train(model, train_loader, configs.epochs_pretrain, configs.learning_rate, configs.momentum,
          configs.model_path, log_dir="runs/pretrain", phase="pretrain")

    print("==> Evaluating pretrained model...")
    evaluate(model, test_loader)

    print("==> Finetuning on same MNIST dataset (simulated)...")
    model.load_state_dict(torch.load(configs.model_path))
    train(model, train_loader, configs.epochs_finetune, configs.learning_rate / 10, configs.momentum,
          log_dir="runs/finetune", phase="finetune")

    print("==> Final evaluation after finetuning...")
    evaluate(model, test_loader)

    print("==> To visualize training, run:tensorboard --logdir=runs")

if __name__ == "__main__":
    main()