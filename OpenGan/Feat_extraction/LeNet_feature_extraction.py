from .Feature_extraction_abs import Feature_extraction_abs

from Modelos import LeNet,LeNet_cac
import torch.nn as nn
import torch.nn.functional as F
import torch
device = 'cuda:0'

class LeNet_feature_extraction(Feature_extraction_abs):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = LeNet(num_classes)
        self.adjust_output()
    
    def adjust_output(self):
        self.model.to(device=device)
        
        
    def forward(self, x):
        return self.model(x)
    
    def classify_features(self, x):
        return self.model.fc3(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            x = F.relu(self.model.conv1(x))
            x = F.avg_pool2d(x, 2)                           # 24x24 -> 12x12
            x = F.relu(self.model.conv2(x))
            x = F.avg_pool2d(x, 2)                           # 8x8 -> 4x4
            x = torch.flatten(x, 1)
            x = F.relu(self.model.fc1(x))
            x = F.relu(self.model.fc2(x))
            x = torch.flatten(x, 1)

        return x
    
    def load_model(self, weights):
        self.model.load_state_dict(weights)

class LeNet_cac_feature_extraction(Feature_extraction_abs):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = LeNet_cac(num_classes)
        self.model.skip_distance = True
        self.adjust_output()
    
    def adjust_output(self):
        self.model.to(device=device)
        
        
    def forward(self, x):
        return self.model(x)
    
    def classify_features(self, x):
        return self.model.classify(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            x = self.model.encoder(x)

        return x
    
    def load_model(self, weights):
        self.model.load_state_dict(weights)
def main():
    pass

if __name__  == "__main__":
    main()