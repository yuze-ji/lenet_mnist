# lenet/data.py

import torch
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size, data_dir="data"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 for LeNet
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader