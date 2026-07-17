

from Utils import fix_random_seed, NOMES
from Datasets.Load_Data_panicum_halfsize import Panicum_halfsize_loader
import torch
import os,sys
device = 'cuda:0'
seed = 42
fix_random_seed(seed)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from Feat_extraction.ResNet18_feature_extraction import ResNet18_feature_extraction,ResNet18_cac_feature_extraction

def cac():
    num_classes = 2
    
    tranform_feat_extractor = NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value
    data = Panicum_halfsize_loader(bs=32)
    
    
    for fold in range(5):
        model = ResNet18_cac_feature_extraction(num_classes=num_classes)
        model.load_model(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum_cac",NOMES.RESNET18.value,f"Fold_{fold}",f"ResNet18_Panicum_cac_fold_{fold}_plantnet.pt")))

        features_dir = os.path.join(NOMES.FEATS_DIR.value,"Panicum_plantnet_cac",NOMES.RESNET18.value,f"Fold_{fold}")

        train_dataloader    = data.load_train(fold,tranform_feat_extractor)
        kkc_val_dataloader  = data.load_kkc_val(fold,tranform_feat_extractor)
        uuc_test_dataloader = data.load_uuc_test(fold,tranform_feat_extractor)
        kkc_test_dataloader = data.load_kkc_test(fold,tranform_feat_extractor)
        test_dataloader     = data.load_test(fold,tranform_feat_extractor)
        uuc_val_dataloader = data.load_uuc_val(fold,tranform_feat_extractor)
        val_dataloader = data.load_val(fold,tranform_feat_extractor)




        if not os.path.exists(features_dir):
            os.makedirs(features_dir, exist_ok=True)
    #torch.save(model.model.state_dict(), features_dir + "modelo.pth")
        model.save_features(train_dataloader,features_dir,"panicum_treino")
        model.save_features(kkc_val_dataloader,features_dir,"kkc_val")
        model.save_features(uuc_test_dataloader,features_dir,"uuc_test")
        model.save_features(kkc_test_dataloader,features_dir,"kkc_test")
        model.save_features(test_dataloader,features_dir,"test")
        model.save_features(val_dataloader,features_dir,"val")
        model.save_features(uuc_val_dataloader,features_dir,"uuc_val")
def no_cac():
    num_classes = 2
    
    tranform_feat_extractor = NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value
    data = Panicum_halfsize_loader(bs=32)
    print(data.root_kkc)
    
    for fold in range(5):
        model = ResNet18_feature_extraction(num_classes=num_classes)
        model.load_model(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum",NOMES.RESNET18.value,f"Fold_{fold}",f"ResNet18_Panicum_fold_{fold}_plantnet.pt")))

        features_dir = os.path.join(NOMES.FEATS_DIR.value,"Panicum_plantnet",NOMES.RESNET18.value,f"Fold_{fold}")

        train_dataloader    = data.load_train(fold,tranform_feat_extractor)
        kkc_val_dataloader  = data.load_kkc_val(fold,tranform_feat_extractor)
        uuc_test_dataloader = data.load_uuc_test(fold,tranform_feat_extractor)
        kkc_test_dataloader = data.load_kkc_test(fold,tranform_feat_extractor)
        test_dataloader     = data.load_test(fold,tranform_feat_extractor)
        uuc_val_dataloader = data.load_uuc_val(fold,tranform_feat_extractor)
        val_dataloader = data.load_val(fold,tranform_feat_extractor)




        if not os.path.exists(features_dir):
            os.makedirs(features_dir, exist_ok=True)
    #torch.save(model.model.state_dict(), features_dir + "modelo.pth")
        model.save_features(train_dataloader,features_dir,"panicum_treino")
        model.save_features(kkc_val_dataloader,features_dir,"kkc_val")
        model.save_features(uuc_test_dataloader,features_dir,"uuc_test")
        model.save_features(kkc_test_dataloader,features_dir,"kkc_test")
        model.save_features(test_dataloader,features_dir,"test")
        model.save_features(val_dataloader,features_dir,"val")
        model.save_features(uuc_val_dataloader,features_dir,"uuc_val")

if __name__ == "__main__":
    cac()