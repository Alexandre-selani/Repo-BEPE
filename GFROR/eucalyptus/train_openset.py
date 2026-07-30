import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.utils import make_grid, save_image
from torchvision.models import AlexNet_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas
import random

from collections import Counter
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, auc
from torchmetrics import AUROC

from model.vanilla_ae import VanillaAE_eucalyptus
from model.wgan import WGAN_GP
from model.utils import to_img, to_4d
from model.classifier import Classifier
from Modelos import LeNet_GFROR,AlexNet_GFROR
from Datasets import Eucalyptus_openset_loader
from Utils import fix_random_seed,NOMES


# train on known classes, both classification and self supervision
def train(G, C, dataloader, optimizer, loss_fn, transformations, device):
    #G.train()
    C.train()
    G.eval()

    ce_losses, ss_losses, train_losses = [], [], []

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x_hat = G(x)
        concat_x = torch.cat((x, x_hat), dim=1)
        ce_loss = loss_fn(C(concat_x)[0], y)

        # note: how to get rid of for loop
        trans_ind = torch.randint(len(transformations), (x.size(0),))
        rand_trans = transformations[trans_ind]
        t_x = torch.stack([t(x[i]) for i,t in enumerate(rand_trans)], dim=0)
        t_x_hat = torch.stack([t(x_hat[i]) for i,t in enumerate(rand_trans)], dim=0)

        concat_t = torch.cat((t_x, t_x_hat), dim=1)
        ss_loss = loss_fn(C(concat_t)[1], trans_ind.to(device))

        loss = 0.8 * ce_loss + 0.2 * ss_loss

        cls_out, ss_out = C(concat_t)

        pred = ss_out.argmax(1)
        acc = (pred == trans_ind.to(device)).float().mean()

        print(acc.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_losses.append(ce_loss.item())
        ss_losses.append(ss_loss.item())
        train_losses.append(loss.item())

    return ce_losses, ss_losses, train_losses

def evaluate_closedSet(G, C, dataloader, optimizer, loss_fn, transformations, device):
    C.eval()
    G.eval()
    
    ce_losses, ss_losses, train_losses = [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            ce_loss = loss_fn(C(concat_x)[0], y)
    
            
    
            ce_losses.append(ce_loss.item())
           

    return ce_losses


import gc
device = "cuda:0"
def main():

    N_FOLDS=5
    save_dir = os.path.join("/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_eucalyptus","AlexNet")
    os.makedirs(save_dir,exist_ok=True)
    gc.collect()

    lr = 0.0001
    epochs = 20
    bs = 20
    num_classes = 2

    weights = AlexNet_Weights.IMAGENET1K_V1
    transforms = weights.transforms()

    # 2. Inicialização do Loader customizado
    data_manager = Eucalyptus_openset_loader(bs=bs)

    transformations = np.array([
            T.RandomRotation(degrees=[90,90]), # deterministic rotation
            T.RandomRotation(degrees=[180,180]),
            T.RandomRotation(degrees=[270,270]),
            T.RandomRotation(degrees=[360,360]), # original input
            T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[90,90])]), # deterministic flip + rotation
            T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[180,180])]),
            T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[270,270])]),
            T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[360,360])]),
        ])

    # 4. Modelo, Critério e Otimizador

    model_name = "AlexNet"

    for fold in range(N_FOLDS):
        gc.collect()
        torch.cuda.empty_cache()
        
        fold_dir = os.path.join(save_dir,f"Fold_{fold}")
        os.makedirs(fold_dir,exist_ok=True)

        classifier = AlexNet_GFROR(num_classes=2,num_transforms=8)
        classifier = classifier.to(device)

        generator = torch.load(f"/home/alexandreselani/Desktop/GFROR/ckpt/ae_eucalyptus/eucalyptus/fold_{fold}.pth",weights_only=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.7)

        train_dataloader,val_kkc_dataloader = data_manager.load_train(fold,transforms), data_manager.load_kkc_val(fold,transforms)


        for epoch in range(epochs):
            
            train_ce_loss, train_ss_loss, train_loss = train(generator, classifier, train_dataloader, optimizer, criterion, transformations, device)
            val_loss = evaluate_closedSet(generator, classifier, val_kkc_dataloader, optimizer, criterion, transformations, device)
            
            #scheduler.step(sum(val_loss)/len(val_loss))
            print('epoch [{}/{}], lr:{:.4f}, train loss:{:.4f}, val_loss: {:.4f}'.format(epoch+1, epochs, optimizer.param_groups[0]['lr'], sum(train_loss)/len(train_loss), sum(val_loss)/len(val_loss)))
            
            
        save_path = os.path.join(fold_dir, f'ckpt.pth')
        torch.save(classifier, save_path)

        del classifier,train_dataloader,val_kkc_dataloader,optimizer,criterion,scheduler

if __name__ == "__main__":
    main()