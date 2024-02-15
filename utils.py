import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize, Compose


def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def train(net, trainloader, optim, epochs, device: str):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    net.train()
    print('11')
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
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
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
