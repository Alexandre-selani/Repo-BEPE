
from torchvision.models import resnet18,ResNet18_Weights
import torch.nn as nn
from .Base_CAC import BaseCACClassifier


def ResNet18(num_classes,weights=None):
    
    model=None 

    if isinstance(weights, str) or weights is None:
        model=resnet18(weights=weights)
    else:
        model = resnet18()
        weights.pop('fc.weight', None)
        weights.pop('fc.bias', None)
        model.load_state_dict(weights,strict=False)

    model.fc = nn.Linear(512,num_classes)

    return model


"""
	Network definition for our proposed CAC open set classifier. 

	Dimity Miller, 2020
"""

import torch

class ResNet18_cac(BaseCACClassifier):
    def __init__(self, num_classes=20, weights=None, skip_distances=False, init_weights=False, **kwargs):
        # Passamos a dimensão de recursos (512) fixa para a ResNet18
        self.weights = weights
        super(ResNet18_cac, self).__init__(
            num_classes=num_classes, 
            feat_dim=512, 
            skip_distances=skip_distances,
            init_weights=init_weights
        )

    def _build_encoder(self) -> nn.Module:
        # Instancia o ResNet18 customizado utilizando a função que você já tinha
        encoder = ResNet18(self.num_classes, self.weights)
        
        # Remove a última camada linear transformando-a em Identidade
        encoder.fc = nn.Identity()
        return encoder