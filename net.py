import torch.nn as nn


class VGG16(nn.Module):

    def __init__(self, cls_num):

        super(VGG16, self).__init__()

        self.linear = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, cls_num)

    def forward(self, x):

        x = self.linear(x)
        x = self.relu(x)

        return self.fc(x)


class Net(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x
