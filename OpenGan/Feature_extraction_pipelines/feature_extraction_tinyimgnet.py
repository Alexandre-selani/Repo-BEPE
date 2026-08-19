

from Utils import fix_random_seed, NOMES
from Datasets.tinyimagenet_loader import TinyImageNet_loader
import torch
import os,sys
device = 'cuda:0'
seed = 42
fix_random_seed(seed)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from Feat_extraction.ResNet18_64x64_feature_extraction import ResNet18_64x64_feature_extraction

num_classes = 20


def main():
    data = TinyImageNet_loader()
    print(data.splits)

    
    for fold in range(5):
        model = ResNet18_64x64_feature_extraction(num_classes=num_classes)
        model.load_model(torch.load(os.path.join(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/Split_{fold}/ResNet18_TinyImageNet_split_{fold}.pt")))

        features_dir = os.path.join(NOMES.FEATS_DIR.value,"Tinyimgnet",NOMES.RESNET18.value,f"Split_{fold}")

        train_dataloader    = data.get_train_loader(fold, data.eval_transforms[fold])
        val_dataloader      = data.get_val_loader(fold, data.eval_transforms[fold])
        test_dataloader     = data.get_test_loader(fold,data.eval_transforms[fold])

        kkc_val_dataloader  = data.get_val_known_loader(fold, data.eval_transforms[fold])
        kkc_test_dataloader = data.get_test_known_loader(fold,data.eval_transforms[fold])

        uuc_val_dataloader = data.get_val_unknown_loader(fold,data.eval_transforms[fold])
        uuc_test_dataloader = data.get_test_unknown_loader(fold,data.eval_transforms[fold])

        if not os.path.exists(features_dir):
            os.makedirs(features_dir, exist_ok=True)
    #torch.save(model.model.state_dict(), features_dir + "modelo.pth")
        model.save_features(train_dataloader,features_dir,"train")
        model.save_features(kkc_val_dataloader,features_dir,"kkc_val")
        model.save_features(uuc_test_dataloader,features_dir,"uuc_test")
        model.save_features(kkc_test_dataloader,features_dir,"kkc_test")
        model.save_features(test_dataloader,features_dir,"test")
        model.save_features(val_dataloader,features_dir,"val")
        model.save_features(uuc_val_dataloader,features_dir,"uuc_val")

if __name__ == "__main__":
    main()