import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
class BaseCACClassifier(nn.Module, ABC):
    def __init__(self, num_classes: int, feat_dim: int, skip_distances: bool = False, init_weights: bool = False):
        """
        Classe abstrata base para classificadores baseados em Class Anchor Clustering (CAC).
        
        Args:
            num_classes (int): Número de classes do problema.
            feat_dim (int): Dimensão da saída do encoder (ex: 512 para ResNet18, 2048 para ResNet50).
            skip_distance (bool): Se True, pula o cálculo de distância no forward.
            init_weights (bool): Se True, inicializa os pesos da rede.
        """
        super(BaseCACClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.skip_distances = skip_distances
        
        # O encoder deve ser construído pela subclasse específica
        self.encoder = self._build_encoder()
        
        # Camada linear que projeta os recursos do encoder para o espaço das classes
        self.classify = nn.Linear(self.feat_dim, self.num_classes)
        
        # Matriz de âncoras invariável pelo gradiente
        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)
        
        if init_weights:
            self._initialize_weights()
            
        self.cuda()

    @abstractmethod
    def _build_encoder(self) -> nn.Module:
        """
        Método abstrato obrigatório. Deve instanciar o backbone 
        e substituir a última camada de classificação por nn.Identity().
        """
        pass

    def forward(self, x):
        batch_size = len(x)

        # Extração de características e achatamento (flatten)
        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)

        if self.skip_distances:
            return outLinear

        outDistance = self.distance_classifier(outLinear)

        return outLinear, outDistance

    def distance_classifier(self, x):
        """Calcula a distância euclidiana de x até cada classe âncora."""
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)

        return dists

    def set_anchors(self, means):
        """Define os centros/médias das âncoras."""
        self.anchors = nn.Parameter(means.double(), requires_grad=False)
        self.cuda()

    def _initialize_weights(self):
        """Inicialização padrão de pesos para os módulos."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def save_model(self,dir:str):
        """Salva modelo e ancoras em um arquivo em dir"""
        torch.save(self.state_dict(),dir)
    
    def predict_by_distance(self,epsilon,distances):
        softmax = torch.nn.Softmax(dim = 1)
        #print(distances)
        softmin = softmax(-distances)
        invScores = 1-softmin
        scores = distances*invScores
        
        min_scores, predicted = torch.min(scores, axis = 1)
        #print(min_scores)
        final_predictions = torch.where(min_scores > epsilon, -1, predicted)
    
        return final_predictions,min_scores,scores
        
