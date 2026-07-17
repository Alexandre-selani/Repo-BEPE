#!/usr/bin/env python
# coding: utf-8

# OpenGAN: Open-Set Recognition via Open Data Generation
# ================
# **Supplemental Material for ICCV2021 Submission**
# 
# 
# In this notebook, we demonstrate cross-domain open-set image classification.
# Specifically, we show how we evaluate the model over diverse open-set images.
# We also visualize the "landscape" of the open-set discriminator. 

# import packages
# ------------------
# 
# Some packages are installed automatically through Anaconda. PyTorch should be also installed.

# In[18]:


from __future__ import print_function, division
import os, random, time, copy

import numpy as np
#import libmr
import pandas as pd
import os.path as path
import scipy.io as sio
from scipy import misc
from scipy import ndimage, signal
import scipy
import pickle
import sys
import math
import matplotlib.pyplot as plt
import PIL.Image
from io import BytesIO
#from skimage import data, img_as_float
#from skimage.measure import compare_ssim as ssim
#from skimage.measure import compare_psnr as psnr

import torch
from torch.utils.data import Dataset, DataLoader

from utils.eval_funcs import *
from utils.dataset_tinyimagenet_3sets import *
from utils.dataset_cifar10 import *
from utils.network_arch_tinyimagenet import *
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
from Classificador import Generator,Discriminator
from Utils import NOMES


from Feat_extraction import ResNet18_feature_extraction
import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)


from Utils_OpenGan import *
from Feat_extraction import ResNet18_feature_extraction

manualSeed = 42
fix_random_seed(manualSeed)


################## set attributes for this project/experiment ##################
# config result folder
exp_dir = '/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Tinyimgnet/OpenGan_tinyimgnetResNet18' # experiment directory, used for reading the init model

modelFlag = "ResNet18"

# This is the directory from which we read a checkpoint.
# We are showing a GAN-fea model, which is trained only on the real, closed-set images.
project_name = 'OpenGan_tinyimgnet' + modelFlag

# set device, which gpu to use.
device ='cpu'
if torch.cuda.is_available(): 
    device='cuda:0'


total_epoch_num = 100 # total number of epoch in training
batch_size = 128    
insertConv = False    
embDimension = 64
isPretrained = False

newsize = (64, 64)


bestEpoch = 99


# Number of channels in the training images. For color images this is 3
nc = 512
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

nClassTotal = 100
nClassCloseset = nClassTotal


lr = 0.0001 # base learning rate

num_epochs = total_epoch_num
torch.cuda.device_count()
torch.cuda.empty_cache()

save_dir = exp_dir
print(save_dir)    
if not os.path.exists(save_dir): os.makedirs(save_dir,exist_ok=True)

log_filename = os.path.join(save_dir, 'train.log')

folder_to_run = os.path.join(exp_dir, project_name)

import gc
class FeatDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data.shape[0]        
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curdata = self.data[idx]        
        return curdata

maiores_rocs = []
melhores_epochs = []

path_to_feat = os.path.join(NOMES.FEATS_DIR.value,"Tinyimgnet",NOMES.RESNET18.value)

#As tinyimagenet's test set is not labeled, it will be used here, given that labels are binary (real or fake). Val set is going to be used in order to assess classification performance
val_closedset  = torch.load(os.path.join(path_to_feat,"test_features.pt"))
val_closedset_dataset = FeatDataset(val_closedset["features"])
dataloader_val_closedset = DataLoader(val_closedset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_openset = torch.load("/home/alexandreselani/Desktop/Features_extraidas/SVHN/ResNet18/SVHN_val_features.pt")
val_openset_dataset = FeatDataset(val_openset["features"])
dataloader_val_openset = DataLoader(val_openset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

maior_roc_iteracao = -1
melhor_epoch = -1

for epoch in range(num_epochs):
    print(f"EPOCH {epoch}")
    gc.collect()
    torch.cuda.empty_cache()

    
    netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf).to(device)
    path_to_D = os.path.join(save_dir,'epoch-{}.DNet'.format(epoch+1))
    
    netD.load_state_dict(torch.load(path_to_D))
    netD.eval()
    

    feat_close_mnist = torch.tensor([]).type(torch.float)
    label_close_mnist = torch.tensor([]).type(torch.float)
    conf_close_mnist = torch.tensor([]).type(torch.float)

    i = 0
    count = 0

    for X in dataloader_val_closedset:    
        X = X.to(device,dtype=torch.float32)
        # y = y.type(torch.long).view(-1).to(device)    
        count += X.shape[0]
        i+=1 
        feats = X.unsqueeze_(-1).unsqueeze_(-1)
        with torch.no_grad():
            predConf = netD(feats)
        predConf = predConf.view(-1,1)
        #print(predConf[0,0])
        conf_close_mnist = torch.cat((conf_close_mnist, predConf.reshape(-1).detach().cpu()), 0)
        
        feats = feats.squeeze()
        #label_close_mnist = torch.cat((label_close_mnist, y.type(torch.float).detach().cpu().reshape(-1,1))) 
            
    conf_close_mnist = conf_close_mnist.detach().cpu().numpy()
# We draw the ROC curve for classifying closed-set and open-set data.
    feat_open_omniglot = torch.tensor([]).type(torch.float)
    label_open_omniglot = torch.tensor([]).type(torch.float)
    conf_open_omniglot = torch.tensor([]).type(torch.float)

    i = 0
    count = 0
    for X in dataloader_val_openset:
        X = X.to(device,dtype=torch.float32)    
        count += X.shape[0]
        i+=1    
        feats = X.unsqueeze_(-1).unsqueeze_(-1)
        predConf = netD(feats)        
        predConf = predConf.view(-1,1).detach()
        conf_open_omniglot = torch.cat((conf_open_omniglot, predConf.reshape(-1).detach().cpu()), 0)
        #print(conf_open_cifar10)
        feats = feats.squeeze()
        feat_open_omniglot = torch.cat((feat_open_omniglot, feats.detach().cpu()))
        #label_open_cifar10 = torch.cat((label_open_cifar10, torch.Tensor(-1))) 
        

    conf_open_omniglot = conf_open_omniglot.detach().cpu().numpy()    

    roc_score, roc_to_plot = evaluate_openset(-conf_close_mnist, -conf_open_omniglot)

    plt.plot(roc_to_plot['fp'], roc_to_plot['tp'])
    plt.grid('on')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC score {:.5f}'.format(roc_score))
            
    print(roc_score)
    if(roc_score>maior_roc_iteracao):
        maior_roc_iteracao=roc_score
        melhor_epoch = epoch+1

maiores_rocs.append(maior_roc_iteracao)
melhores_epochs.append(melhor_epoch)

for iter, roc in enumerate(maiores_rocs):
    print(f"Maior roc na Iteração {iter}: {roc} (epoch {melhores_epochs[iter]})")


print(f"roc media = {np.array(maiores_rocs).mean()}")


