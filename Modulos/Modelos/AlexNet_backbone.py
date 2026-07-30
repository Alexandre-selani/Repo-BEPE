
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


class AlexNet_GFROR(nn.Module):
    """AlexNet adaptado para o GFROR: 6 canais de entrada (x + x_hat).

    Saidas:
        classification_out: (batch, num_classes)
        transformation_out: (batch, num_transforms)
    """

    def __init__(self, num_classes=10, num_transforms=10):
        super().__init__()
        # Bloco de features baseado no AlexNet
        # Primeira camada adaptada para 6 canais (concatenação x + x_hat)
        self.features = nn.Sequential(
            # Conv1 — 6 canais de entrada
            nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        # Classificador compartilhado (sem a última camada linear)
        self.classification = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        
        # Cabeça de transformação
        self.transformation = nn.Sequential(
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_transforms)
                )
       

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        classification_out = self.classification(x)
        transformation_out = self.transformation(x)
        return classification_out, transformation_out


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