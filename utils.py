import glob
import re
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Compose


def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    image_key = 'image' if 'image' in batch else 'img'
    batch[image_key] = [transforms(img) for img in batch[image_key]]
    return batch


def train(net, trainloader, optim, epochs, patience, device: Union[str, torch.device]):
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


def test(net, testloader, device: Union[str, torch.device]):
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


def increment_path(dst_path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    dst_path = Path(dst_path)  # os-agnostic
    if dst_path.exists() and not exist_ok:
        suffix = dst_path.suffix
        dst_path = dst_path.with_suffix('')
        dirs = glob.glob(f"{dst_path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % dst_path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 1  # increment number
        dst_path = Path(f"{dst_path}{sep}{n}{suffix}")  # update path
    _dir = dst_path if dst_path.suffix == '' else dst_path.parent  # directory
    if not _dir.exists() and mkdir:
        _dir.mkdir(parents=True, exist_ok=True)  # make directory
    return dst_path
