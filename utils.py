from typing import Union

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Compose


def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    image_key = 'image' if 'image' in batch else 'img'
    batch[image_key] = [transforms(img) for img in batch[image_key]]
    return batch


def train(net, trainloader, optim, epochs, patience, device: str):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            image_key = 'image' if 'image' in batch else 'img'
            images, labels = batch[image_key].to(device), batch['label'].to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            image_key = 'image' if 'image' in data else 'img'
            images, labels = data[image_key].to(device), data['label'].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
