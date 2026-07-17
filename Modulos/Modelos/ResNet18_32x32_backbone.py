
from torchvision.models import resnet18,ResNet18_Weights
import torch.nn as nn


def ResNet18_32x32(num_classes,weights=None):
    model=resnet18(weights=weights)
    model.fc = nn.Linear(512,num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
