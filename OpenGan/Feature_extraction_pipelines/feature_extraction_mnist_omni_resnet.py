
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
from Feat_extraction.ResNet18_feature_extraction import ResNet18_feature_extraction
from Utils import fix_random_seed, NOMES
from Datasets import Mnist_omni_loader
import torch

device = 'cuda:0'
seed = 42
fix_random_seed(seed)

num_classes = 10
model = ResNet18_feature_extraction(num_classes=num_classes)
model.load_model(torch.load("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"))



features_dir = os.path.join(NOMES.FEATS_DIR.value,NOMES.MNIST_OMNI.value,NOMES.RESNET18.value)

data = Mnist_omni_loader(bs=256,transform=NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value)

mnist_train_dataloader   =  data.load_train()
mnist_val_dataloader     =  data.load_mnist_val()
mnist_test_dataloader    =  data.load_mnist_test()
omniglot_val_dataloader  =  data.load_omni_val()
omniglot_test_dataloader =  data.load_omni_test()

if not os.path.exists(features_dir):
    os.makedirs(features_dir, exist_ok=True)
#torch.save(model.model.state_dict(), features_dir + "modelo.pth")
model.save_features(mnist_train_dataloader,features_dir,"mnist_treino")
model.save_features(mnist_val_dataloader,features_dir,"mnist_val")
model.save_features(mnist_test_dataloader,features_dir,"mnist_test")
model.save_features(omniglot_val_dataloader,features_dir,"omni_val")
model.save_features(omniglot_test_dataloader,features_dir,"omni_test")