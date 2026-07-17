#!/usr/bin/env python
# coding: utf-8

# OpenGAN: Open-Set Recognition via Open Data Generation
# ================
# **Supplemental Material for ICCV2021 Submission**
# 
# 
# In this notebook, we demonstrate how we train the GAN-fea model on the TinyImageNet train-set, providing the closed-set images.

# import packages
# ------------------
# 
# Some packages are installed automatically through Anaconda. PyTorch should be also installed.

# In[17]:


from __future__ import print_function, division
import os, random, time, copy
from skimage import io, transform
import numpy as np
#import libmr
import pandas as pd
import os.path as path
import sys
import math
import matplotlib.pyplot as plt

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
from utils.dataset_tinyimagenet import *
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(PARENT_DIR)

from Utils_OpenGan import FeatDataset
from Classificador import Discriminator,Generator,weights_init
from Utils import fix_random_seed,NOMES
import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)


manualSeed = 42
fix_random_seed(manualSeed)

def test(netdD,dataloader_test_closedset,dataloader_uuc_test):
    netD.eval()
    feat_close_tinyimagenet = torch.tensor([]).type(torch.float)
    label_close_tinyimagenet = torch.tensor([]).type(torch.float)
    conf_close_tinyimagenet = torch.tensor([]).type(torch.float)
    i = 0
    count = 0
    for X,y in dataloader_test_closedset:    
        X = X.to(device,dtype=torch.float32)
    # y = y.type(torch.long).view(-1).to(device)    
        count += X.shape[0]
        i+=1 
        feats = X.unsqueeze_(-1).unsqueeze_(-1)
        predConf = netD(feats)
        predConf = predConf.view(-1,1)
        conf_close_tinyimagenet = torch.cat((conf_close_tinyimagenet, predConf.reshape(-1).detach().cpu()), 0)
        
        feats = feats.squeeze()
        #label_close_tinyimagenet = torch.cat((label_close_tinyimagenet, y.type(torch.float).detach().cpu().reshape(-1,1))) 
            
    conf_close_tinyimagenet = conf_close_tinyimagenet.detach().cpu().numpy()
# We draw the ROC curve for classifying closed-set and open-set data.
    feat_open_cifar10 = torch.tensor([]).type(torch.float)
    label_open_cifar10 = torch.tensor([]).type(torch.float)
    conf_open_cifar10 = torch.tensor([]).type(torch.float)

    i = 0
    count = 0
    for X,y in dataloader_uuc_test:
        X = X.to(device,dtype=torch.float32)    
        count += X.shape[0]
        i+=1    
        feats = X.unsqueeze_(-1).unsqueeze_(-1)
        predConf = netD(feats)        
        predConf = predConf.view(-1,1).detach()
        conf_open_cifar10 = torch.cat((conf_open_cifar10, predConf.reshape(-1).detach().cpu()), 0)
        #print(conf_open_cifar10)
        feats = feats.squeeze()
        feat_open_cifar10 = torch.cat((feat_open_cifar10, feats.detach().cpu()))
        #label_open_cifar10 = torch.cat((label_open_cifar10, torch.Tensor(-1))) 
        

    conf_open_cifar10 = conf_open_cifar10.detach().cpu().numpy()    

    roc_score, roc_to_plot = evaluate_openset(-conf_close_tinyimagenet, -conf_open_cifar10)

    return roc_score
# Setup config parameters
#  -----------------
#  
#  There are several things to setup, like which GPU to use, where to read images and save files, etc. Please read and understand this. By default, you should be able to run this script smoothly by changing nothing.

modelFlag = NOMES.RESNET18.value

project_name = 'OpenGan_Panicum_plantnet' + modelFlag   # we save all the checkpoints in this directory

# set device, which gpu to use.
device ='cpu'
if torch.cuda.is_available(): 
    device='cuda:0'


total_epoch_num = 500 # total number of epoch in training
batch_size = 15

newsize = (64, 64)

path_to_feats = os.path.join(NOMES.FEATS_DIR.value,"Panicum_plantnet",modelFlag) # the path to cached off-the-shelf features

# Number of channels in the training images. For color images this is 3
nc = 512

# Size of z latent vector (i.e. size of generator input)
nz = 90

# Size of feature maps in generator
ngf = 80

# Size of feature maps in discriminator
ndf = 80

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


nClassTotal = 2
nClassCloseset = nClassTotal


lr = 0.0005 # learning rate

num_epochs = total_epoch_num
torch.cuda.device_count()
torch.cuda.empty_cache()

exp_dir = os.path.join("/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Panicum_plantnet/",modelFlag)

print(path_to_feats)


for iteration in range(5):
    save_discr_dir = os.path.join(exp_dir,f"Fold+{iteration}")
    os.makedirs(save_discr_dir,exist_ok=True)

    netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf).to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    netG.apply(weights_init)


    panicum_train = torch.load(os.path.join(path_to_feats,f'Fold_{iteration}',"panicum_treino_features.pt"))
    panicum_train_features = panicum_train

    trainset_closeset = FeatDataset(data=panicum_train_features)
    dataloader = DataLoader(trainset_closeset, batch_size=batch_size, shuffle=True, num_workers=1)

    uuc_test  = torch.load(os.path.join(path_to_feats,f'Fold_{iteration}',"uuc_test_features.pt"))
    uuc_test_dataset = FeatDataset(uuc_test)
    dataloader_uuc_test = DataLoader(uuc_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    kkc_test  = torch.load(os.path.join(path_to_feats,f'Fold_{iteration}',"kkc_test_features.pt"))
    kkc_test_dataset = FeatDataset(kkc_test)
    dataloader_test_closedset = DataLoader(kkc_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr/1.5, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch

    best_roc_in_fold = -1
    best_epoch_in_fold = -1
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data,y) in enumerate(dataloader, 0):
                ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.train()
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            real_cpu = real_cpu.view(real_cpu.size(0), nc, 1, 1).to(device,dtype=torch.float32)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device,dtype=torch.float32)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            #print(output.shape)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        ###test discriminator
        epoch_roc = test(netD,dataloader_test_closedset=dataloader_test_closedset,dataloader_uuc_test=dataloader_uuc_test)
        print(f"ROC: {epoch_roc}")

        if(epoch_roc > best_roc_in_fold ):
            best_roc_in_fold=epoch_roc
            cur_model_wts = copy.deepcopy(netD.state_dict())
            path_to_save_paramOnly = os.path.join(save_discr_dir, 'best_epoch.DNet'.format(epoch+1))
            torch.save(cur_model_wts, path_to_save_paramOnly)

            

        iters += 1
            
    del criterion,optimizerD,optimizerG,netD,netG

    # ## drawing the error curves

    # In[28]:


    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(exp_dir,f'learningCurves_{iteration}.png'), bbox_inches='tight',transparent=True)
    # plt.show()




