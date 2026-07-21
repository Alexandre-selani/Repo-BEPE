
from torchvision.models import alexnet
import torch.nn as nn
import torch
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


class AlexNetFeaturizer(nn.Module):
    """Wrapper que retorna (logits, features) no forward, similar ao ResNet18Featurizer.

    Usa os mesmos nomes de camadas do AlexNet original (features, avgpool, classifier)
    para compatibilidade total de state_dict com modelos treinados via funcao Alexnet().
    """

    def __init__(self, num_classes=10):
        super().__init__()
        backbone = alexnet()

        # Blocos de features com os MESMOS nomes do AlexNet original
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Classificador completo com a ultima camada substituida
        self.classifier = backbone.classifier
        self.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Percorre classifier[0..5] para extrair features 4096-d
        for i in range(6):
            x = self.classifier[i](x)
        feats = x

        # classifier[6] = camada de classificacao final
        logits = self.classifier[6](feats)
        return logits, feats

    def getPerClassWeights(self):
        """Obtém os pesos da última camada (classifier[6])."""
        with torch.no_grad():
            return self.classifier[6].weight.detach()