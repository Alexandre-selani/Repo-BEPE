from Feat_extraction.ResNet18_feature_extraction import ResNet18_feature_extraction
from Feat_extraction.LeNet_feature_extraction import LeNet_feature_extraction
from torchvision.datasets import MNIST, Omniglot
from torch.utils.data import DataLoader,Subset,ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim

from Utils import fix_random_seed, NOMES
from Datasets.Load_Data_datasets_secundarios import Tinyimagenet_loader
import torch
import os
device = 'cuda:0'
seed = 42
fix_random_seed(seed)



def main():
    num_classes = 200
    model = ResNet18_feature_extraction(num_classes=num_classes)
    model.load_model(torch.load(NOMES.RESNET18_TINY_IMAGE_NET.value))

    features_dir = os.path.join(NOMES.FEATS_DIR.value,NOMES.TINY_IMAGE_NET.value,NOMES.RESNET18.value)
    
    data = Tinyimagenet_loader(bs=256)

    
    train_dataloader   =  data.load_train(transform=NOMES.TINY_IMAGE_NET_RESNET18_VAL_TEST_TRANSFORMS.value)
    val_dataloader  =  data.load_val(transform=NOMES.TINY_IMAGE_NET_RESNET18_VAL_TEST_TRANSFORMS.value)
    test_dataloader =  data.load_test(transform=NOMES.TINY_IMAGE_NET_RESNET18_VAL_TEST_TRANSFORMS.value)

    if not os.path.exists(features_dir):
        os.makedirs(features_dir, exist_ok=True)
    #torch.save(model.model.state_dict(), features_dir + "modelo.pth")
    model.save_features(train_dataloader,features_dir,"treino")
    model.save_features(val_dataloader,features_dir,"val")
    model.save_features(test_dataloader,features_dir,"test")
    

if __name__ == "__main__":
    main()