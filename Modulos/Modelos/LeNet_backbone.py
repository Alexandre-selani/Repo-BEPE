import torch
import torch.nn as nn
import torch.nn.functional as F
from .Base_CAC import BaseCACClassifier

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # C1
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # 28x28 -> 24x24
        # C3
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     # 12x12 -> 8x8

        # F6
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)                           # 24x24 -> 12x12
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)                           # 8x8 -> 4x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def LeNet(num_classes):
    model = LeNet5(num_classes)
    return model

class LeNet_cac(BaseCACClassifier):
    def __init__(self, num_classes=20, skip_distances=False, init_weights=False, **kwargs):
        super(LeNet_cac, self).__init__(
            num_classes=num_classes, 
            feat_dim=84, 
            skip_distances=skip_distances, 
            init_weights=init_weights
        )

    def _build_encoder(self) -> nn.Module:
        # Instancia o LeNet5
        encoder = LeNet5(self.num_classes)

        # Remove a última camada linear (fc3) transformando-a em Identidade
        encoder.fc3 = nn.Identity()
        return encoder

    