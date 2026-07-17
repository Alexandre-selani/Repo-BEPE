from Feat_extraction.Feature_extraction_abs import Feature_extraction_abs

from torchvision.models import alexnet
import torch.nn as nn
import torch
device = 'cuda:0'

class AlexNet_feature_extraction(Feature_extraction_abs):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = alexnet()
        self.adjust_output()
    
    def adjust_output(self):
        self.model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=self.num_classes
        )
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)
    
    def classify_features(self, x):
        return self.model.classifier[6](x)
    
    def load_model(self, weights):
        self.model.load_state_dict(weights)
        
    def extract_features(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.classifier[:6](x)

        return x
    
def main():
    extractor = AlexNet_feature_extraction(num_classes=10)
    input_fake = torch.randn(64, 3, 224, 224).to(device=device)
    features = extractor.extract_features(input_fake)

    print(f"Dimensão das features da AlexNet: {features}") 
    # Resultado esperado: torch.Size([1, 4096])

if __name__  == "__main__":
    main()