
from torchvision.models import alexnet
import torch.nn as nn
from .Base_CAC import BaseCACClassifier

def Alexnet(num_classes,weights=None):
    model = alexnet(weights=weights)

    model.classifier[6] = nn.Linear(
    in_features=4096,
    out_features=num_classes
    )

    return model

class AlexNet_cac(BaseCACClassifier):
    def __init__(self, num_classes=20, weights=None, skip_distances=False, init_weights=False, **kwargs):
        # Passamos a dimensão de recursos (512) fixa para a ResNet18
        self.weights = weights
        super(AlexNet_cac, self).__init__(
            num_classes=num_classes, 
            feat_dim=4096, 
            skip_distances=skip_distances,
            init_weights=init_weights
        )

    def _build_encoder(self) -> nn.Module:
        # Instancia o ResNet18 customizado utilizando a função que você já tinha
        encoder = Alexnet(self.num_classes, self.weights)
        
        # Remove a última camada linear transformando-a em Identidade
        encoder.classifier[6] = nn.Identity()
        return encoder