# lenet/train.py with TensorBoard and CSV logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
import os

def train(model, train_loader, epochs, lr, momentum, model_path=None, log_dir="runs/exp", phase="pretrain"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(f"==> Starting {phase}...")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    log_file = os.path.join(log_dir, f"{phase}_log.csv")

    with open(log_file, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Epoch", "Loss"])

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"{phase.capitalize()} Epoch {epoch+1}/{epochs}")):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 0:
                    step = epoch * len(train_loader) + i
                    writer.add_scalar(f"Loss/{phase}", loss.item(), step)

            avg_loss = running_loss / len(train_loader)
            csv_writer.writerow([epoch + 1, avg_loss])
            print(f"{phase.capitalize()} Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    if model_path:
        torch.save(model.state_dict(), model_path)

    writer.close()