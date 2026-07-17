from Feat_extraction.ResNet18_feature_extraction import ResNet18_feature_extraction
from torchvision.models import ResNet18_Weights
from Utils import fix_random_seed, NOMES
from Datasets.Load_Data_panicum import Panicum_loader
import torch
import os
device = 'cuda:0'
seed = 42
fix_random_seed(seed)



def main():
    num_classes = 2
    
    weights = ResNet18_Weights.IMAGENET1K_V1
    data = Panicum_loader(bs=32,transform=weights.transforms())

    
    for fold in range(5):
        model = ResNet18_feature_extraction(num_classes=num_classes)
        model.load_model(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum",NOMES.RESNET18.value,f"Fold_{fold}",f"ResNet18_Panicum_fold_{fold}.pt")))

        features_dir = os.path.join(NOMES.FEATS_DIR.value,"Panicum",NOMES.RESNET18.value,f"Fold_{fold}")

        train_dataloader    = data.load_train(fold)
        kkc_val_dataloader  = data.load_kkc_val(fold)
        uuc_test_dataloader = data.load_uuc_test(fold)
        kkc_test_dataloader = data.load_kkc_test(fold)
        test_dataloader     = data.load_test(fold)




        if not os.path.exists(features_dir):
            os.makedirs(features_dir, exist_ok=True)
    #torch.save(model.model.state_dict(), features_dir + "modelo.pth")
        model.save_features(train_dataloader,features_dir,"panicum_treino")
        model.save_features(kkc_val_dataloader,features_dir,"kkc_val")
        model.save_features(uuc_test_dataloader,features_dir,"uuc_test")
        model.save_features(kkc_test_dataloader,features_dir,"kkc_test")
        model.save_features(test_dataloader,features_dir,"test")

if __name__ == "__main__":
    main()