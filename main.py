# main.py

import argparse
import os

import torch

from lenet import config
from lenet.lenet5 import LeNet5
from lenet.mnist import get_mnist_loaders
from lenet.train import train
from lenet.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="LeNet-5 two-stage (pretrain + finetune) training on MNIST")
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--epochs-pretrain", type=int, default=config.epochs_pretrain)
    parser.add_argument("--epochs-finetune", type=int, default=config.epochs_finetune)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--momentum", type=float, default=config.momentum)
    parser.add_argument("--data-dir", default=config.data_dir)
    parser.add_argument("--checkpoint-dir", default=config.checkpoint_dir)
    parser.add_argument("--log-dir", default=config.log_dir)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    pretrain_path = os.path.join(args.checkpoint_dir, "lenet_pretrained.pth")
    finetune_path = os.path.join(args.checkpoint_dir, "lenet_finetuned.pth")

    train_loader, test_loader = get_mnist_loaders(args.batch_size, data_dir=args.data_dir)

    print("==> Pretraining LeNet5 on MNIST...")
    model = LeNet5()
    train(model, train_loader, args.epochs_pretrain, args.lr, args.momentum,
          pretrain_path, log_dir=os.path.join(args.log_dir, "pretrain"), phase="pretrain")

    print("==> Evaluating pretrained model...")
    evaluate(model, test_loader)

    print("==> Finetuning on same MNIST dataset (simulated)...")
    model.load_state_dict(torch.load(pretrain_path))
    train(model, train_loader, args.epochs_finetune, args.lr / 10, args.momentum,
          finetune_path, log_dir=os.path.join(args.log_dir, "finetune"), phase="finetune")

    print("==> Final evaluation after finetuning...")
    evaluate(model, test_loader)

    print(f"==> To visualize training, run: tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()
