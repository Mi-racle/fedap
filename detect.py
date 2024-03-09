import numpy as np
import torch

from net import resnet18

img = torch.rand(1, 3, 128, 128)

model = resnet18(pretrained=False, in_channels=3, num_classes=53)
model.load_state_dict(torch.load('best.pt'))

output = model(img)
print(output)
