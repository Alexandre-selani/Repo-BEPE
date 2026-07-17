

from Utils import fix_random_seed, NOMES
from Datasets import Eucalyptus_openset_loader
import torch
import os,sys
device = 'cuda:0'
seed = 42
fix_random_seed(seed)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
from torchvision.models import AlexNet_Weights
from Feat_extraction.AlexNet_feature_extraction import AlexNet_feature_extraction

dataset = "dataset-1"

def no_cac():
    num_classes = 2
    
    weights = AlexNet_Weights.IMAGENET1K_V1

    data = Eucalyptus_openset_loader(bs=32)
    
    
    for fold in range(5):
        model = AlexNet_feature_extraction(num_classes=num_classes)
        model.load_model(torch.load(os.path.join("/home/alexandreselani/Desktop/Eucalyptus/OpenSet/Models/dataset-1",f"AlexNet_fold_{fold}.pt")))

        features_dir = os.path.join(NOMES.FEATS_DIR.value,"Eucalyptus",dataset,"AlexNet",f"Fold_{fold}")

        train_dataloader    = data.load_train(fold,weights.transforms())
        kkc_val_dataloader  = data.load_kkc_val(fold,weights.transforms())
        uuc_test_dataloader = data.load_uuc_test(fold,weights.transforms())
        kkc_test_dataloader = data.load_kkc_test(fold,weights.transforms())
        test_dataloader     = data.load_test(fold,weights.transforms())
        uuc_val_dataloader = data.load_uuc_val(fold,weights.transforms())
        val_dataloader = data.load_val(fold,weights.transforms())




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
    no_cac()