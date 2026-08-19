from .Feature_extraction_abs import Feature_extraction_abs

from torchvision.models import resnet18
import torch.nn as nn
import torch
device = 'cuda:0'

class ResNet18_64x64_feature_extraction(Feature_extraction_abs):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = resnet18()
        self.adjust_output()
    
    def adjust_output(self):
        self.model.to(device=device)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes).to(device=device)
        
    def forward(self, x):
        return self.model(x)
    
    def classify_features(self, x):
        return self.model.fc(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)

        return x
    
    def load_model(self, weights):
        self.model.load_state_dict(weights)
