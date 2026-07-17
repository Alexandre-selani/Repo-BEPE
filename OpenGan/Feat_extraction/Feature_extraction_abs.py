from torchvision.models import resnet18
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
device='cuda:0'
class Feature_extraction_abs(nn.Module, ABC):
    def __init__(self,num_classes:int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """Forward padrão"""
        pass

    @abstractmethod
    def extract_features(self, x):
        """Retorna features antes da camada de classificação"""
        pass
    
    @abstractmethod
    def classify_features(self,x):
        """Recebe features para classificar"""
        pass
    @abstractmethod
    def adjust_output(self):
        """Ajusta a camada de saída para num_classes"""
        pass
    
    def save_features(self,dataloader,save_dir,features_name,labels_originais=None):
        all_features = []
        all_labels = []

        
        
        for X,y in dataloader:
            X = X.to(device)
            feat = self.extract_features(X)
            feat = feat.float()
            all_features.append(feat.cpu())
            all_labels.append(y.float())

        #print(all_labels)
        data_to_save = {
        'features': torch.cat(all_features, dim=0),
        'labels': torch.cat(all_labels, dim=0)
        }
      
        if labels_originais is not None:
            data_to_save["labels_originais"] = torch.cat([t.unsqueeze(0) for t in labels_originais], dim=0)

        #print(data_to_save["labels_originais"])
        # Cria a pasta se não existir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Salva o arquivo
        file_path = os.path.join(save_dir, f"{features_name}_features.pt")
        torch.save(data_to_save, file_path)
        print(f"Salvo: {file_path} | Shape: {data_to_save['features'].shape}")
    
    @abstractmethod
    def load_model(self,weights):
        pass
