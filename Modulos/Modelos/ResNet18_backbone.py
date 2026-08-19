
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .Base_CAC import BaseCACClassifier


def ResNet18(num_classes, weights=None):
    
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


class ResNet18Featurizer(nn.Module):
    """Wrapper que retorna (logits, features) no forward, similar ao LeNetFeaturizer.

    Usa os mesmos nomes de camadas do ResNet18 original (conv1, bn1, layer1, etc.)
    para compatibilidade total de state_dict com modelos treinados via funcao ResNet18().
    """

    def __init__(self, num_classes=10, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        # Blocos de features com os MESMOS nomes do ResNet18 original
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Classificador linear (substitui o fc original)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)                          # (batch, 512, 1, 1)
        feats = torch.flatten(x, 1)                  # (batch, 512)
        logits = self.fc(feats)                      # (batch, num_classes)
        return logits, feats

    def getPerClassWeights(self):
        """Obtém os pesos da última camada (classificador fc)."""
        with torch.no_grad():
            return self.fc.weight.detach()


class ResNet18_tinyimgnet(nn.Module):
    """ResNet18 com stem estilo CIFAR/TinyImageNet e cabeca com dropout.

    O stem original (conv 7x7/stride2 + maxpool/stride2) foi desenhado para entradas 224x224;
    em imagens 64x64 ele reduz o mapa de features quase a nada antes da layer4, entao aqui ele
    e trocado por conv 3x3/stride1 sem maxpool (mesma ideia usada para CIFAR).
    """

    def __init__(self, num_classes=20, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = nn.Identity()

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18_GFROR(nn.Module):
    """ResNet18 adaptado para o GFROR: 6 canais de entrada (x + x_hat) e imagens 32x32.

    Saidas:
        classification_out: (batch, num_classes)
        transformation_out: (batch, num_transforms)
    """

    def __init__(self, num_classes=10, num_transforms=10, weights=None):
        super().__init__()
        backbone = resnet18(weights=weights)

        # Conv1 — 6 canais de entrada (concatenação x + x_hat), 32x32
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        self.layer4_class = backbone.layer4
        self.avgpool_class = backbone.avgpool       

        self.layer4_trans = backbone.layer4
        self.avgpool_trans = backbone.avgpool             # adaptive_avg_pool2d(1)


        self.classification = nn.Linear(512, num_classes)
        self.transformation = nn.Linear(512, num_transforms)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_class = self.layer4_class(x)
        x_class = self.avgpool_class(x_class)
        
        x_trans = self.layer4_trans(x)
        x_trans = self.avgpool_trans(x_trans)   
                                    
        x_class = torch.flatten(x_class, 1)
        x_trans = torch.flatten(x_trans,1)                          


        classification_out = self.classification(x_class)
        transformation_out = self.transformation(x_trans)
        return classification_out, transformation_out