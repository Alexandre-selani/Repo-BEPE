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

class LeNet_GFROR(nn.Module):
    """LeNet adaptado para o GFROR: 6 canais de entrada (x + x_hat) e imagens 32x32.

    Saidas:
        classification_out: (batch, num_classes)
        transformation_out: (batch, num_transforms)
    """

    def __init__(self, num_classes=10, num_transforms=10):
        super().__init__()
        # C1 — 6 canais de entrada (concatenação x + x_hat), 32x32 -> 28x28
        self.conv1 = nn.Conv2d(6, 6, kernel_size=5)
        # C3 — 28x14 -> 10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # apos dois avg_pool(2): 32 -> 28 -> 14 -> 10 -> 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.classification = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

        self.transformation = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_transforms)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))                         # 32x32 -> 28x28
        x = F.avg_pool2d(x, 2)                            # 28x28 -> 14x14
        x = F.relu(self.conv2(x))                         # 14x14 -> 10x10
        x = F.avg_pool2d(x, 2)                            # 10x10 -> 5x5
        x = torch.flatten(x, 1)                           # (batch, 16*5*5)
        x = F.relu(self.fc1(x))                           # (batch, 120)

        classification_out = self.classification(x)
        transformation_out = self.transformation(x)
        return classification_out, transformation_out


class LeNetFeaturizer(nn.Module):
    """Wrapper que retorna (logits, features) no forward, similar ao ResNetFeaturizer.

    Usa os mesmos nomes de camadas do LeNet5 para compatibilidade de state_dict.
    """

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
        feats = F.relu(self.fc2(x))
        logits = self.fc3(feats)
        return logits, feats

    def getPerClassWeights(self):
        """Obtém os pesos da última camada (classificador fc3)."""
        with torch.no_grad():
            return self.fc3.weight.detach()