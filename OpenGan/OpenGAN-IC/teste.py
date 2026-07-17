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
from skimage import io, transform
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
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from utils.eval_funcs import *
from utils.dataset_tinyimagenet_3sets import *
from utils.dataset_cifar10 import *
from utils.network_arch_tinyimagenet import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PARENT_DIR)
from Feat_extraction import ResNet18_feature_extraction
import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)


from Utils_OpenGan import *
from Feat_extraction import AlexNet_feature_extraction

manualSeed = 42
fix_random_seed(manualSeed)


# Setup config parameters
#  -----------------
#  
#  There are several things to setup, like which GPU to use, where to read images and save files, etc. Please read and understand this. By default, you should be able to run this script smoothly by changing nothing.

# In[19]:


################## set attributes for this project/experiment ##################
# config result folder
exp_dir = '/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Mnist_omni/' # experiment directory, used for reading the init model

modelFlag = 'AlexNet'

# This is the directory from which we read a checkpoint.
# We are showing a GAN-fea model, which is trained only on the real, closed-set images.
project_name = 'OpenGan_Mnist_omni' + modelFlag

# set device, which gpu to use.
device ='cpu'
if torch.cuda.is_available(): 
    device='cuda:0'


total_epoch_num = 300 # total number of epoch in training
batch_size = 256    
insertConv = False    
embDimension = 64
isPretrained = False

newsize = (64, 64)



# Number of channels in the training images. For color images this is 3
nc = 4096
# Size of z latent vector (i.e. size of generator input)
nz = 120
# Size of feature maps in generator
ngf = 128
# Size of feature maps in discriminator
ndf = 128
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1




nClassTotal = 10
nClassCloseset = nClassTotal


lr = 0.0001 # base learning rate

num_epochs = total_epoch_num
torch.cuda.device_count()
torch.cuda.empty_cache()

save_dir = exp_dir
print(save_dir)    
if not os.path.exists(save_dir): os.makedirs(save_dir,exist_ok=True)

log_filename = os.path.join(save_dir, 'train.log')


encoder_num_layers = 18
embDimension = -1 # negative testues meaning there is no intermediate layers; otherwise there is an addition layer for dimensionality reduction
  # the model epoch as the SOTA classification model on TinyImageNet training data.



encoder = AlexNet_feature_extraction(num_classes=nClassCloseset)
#clsModel = TinyImageNet_ClsNet(nClass=nClassTotal, layerList=(512, ))

folder_to_run = os.path.join(exp_dir, project_name)
#path_to_clsnet = os.path.join(folder_to_run, 'epoch-{}_clsnet.paramOnly'.format(bestEpoch))

path_to_encoder = "/home/alexandreselani/Desktop/OpenGan/Features_extraidas/Mnist_Omni/modelo.pth"

#encoder.model.load_state_dict(torch.load(path_to_encoder))
#clsModel.load_state_dict(torch.load(path_to_clsnet))

encoder.cuda()
encoder.eval()
encoder.to(device)
#clsModel.cuda()
#clsModel.eval()
#clsModel.to(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)     
        

class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=512):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d( self.nz, self.ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d( self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d( self.ngf * 2, self.ngf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d( self.ngf*4, self.nc, 1, 1, 0, bias=True),
            #nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=512, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf*8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor

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

path_to_feat = f"/home/alexandreselani/Desktop/OpenGan/Features_extraidas/Mnist_Omni/"

mnist_test_closedset  = torch.load(os.path.join(path_to_feat,"mnist_test_features.pt"))
mnist_test_closedset_dataset = FeatDataset(mnist_test_closedset["features"])
dataloader_test_closedset = DataLoader(mnist_test_closedset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

omniglot_test_openset = torch.load(os.path.join(path_to_feat,"omni_test_features.pt"))
omniglot_test_openset_dataset = FeatDataset(omniglot_test_openset["features"])
dataloader_test_openset = DataLoader(omniglot_test_openset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

maior_roc_iteracao = -1


best_epoch = 38



gc.collect()
torch.cuda.empty_cache()


netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf).to(device)


path_to_D = os.path.join(save_dir,project_name,'epoch-{}.DNet'.format(best_epoch))

netD.load_state_dict(torch.load(path_to_D))
netD.eval()


feat_close_mnist = torch.tensor([]).type(torch.float)
label_close_mnist = torch.tensor([]).type(torch.float)
conf_close_mnist = torch.tensor([]).type(torch.float)

i = 0
count = 0

for X in dataloader_test_closedset:    
    X = X.to(device,dtype=torch.float32)
    # y = y.type(torch.long).view(-1).to(device)    
    count += X.shape[0]
    i+=1 
    feats = X.unsqueeze_(-1).unsqueeze_(-1)
    with torch.no_grad():
        predConf = netD(feats)
    predConf = predConf.view(-1,1)
    print(predConf[0,0])
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
for X in dataloader_test_openset:
    X = X.to(device,dtype=torch.float32)    
    count += X.shape[0]
    i+=1    
    feats = X.unsqueeze_(-1).unsqueeze_(-1)
    with torch.no_grad():
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
    

maiores_rocs.append(maior_roc_iteracao)


for iter, roc in enumerate(maiores_rocs):
    print(f"Maior roc na Iteração {iter}: {roc} (epoch {melhores_epochs[iter]})")


print(f"roc media = {np.array(maiores_rocs).mean()}")


