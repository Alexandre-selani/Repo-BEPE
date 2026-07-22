import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision.utils import make_grid, save_image
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

from model.vanilla_ae import VanillaAE
from model.wgan import WGAN_GP
from model.utils import to_img, to_4d
from model.classifier import Classifier
from Modelos import LeNet_GFROR
from Datasets import Mnist_omni_loader
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
    
            # note: how to get rid of for loop
            trans_ind = torch.randint(len(transformations), (x.size(0),))
            rand_trans = transformations[trans_ind]
            t_x = torch.stack([t(x[i]) for i,t in enumerate(rand_trans)], dim=0)
            t_x_hat = torch.stack([t(x_hat[i]) for i,t in enumerate(rand_trans)], dim=0)
    
            concat_t = torch.cat((t_x, t_x_hat), dim=1)
            ss_loss = loss_fn(C(concat_t)[1], trans_ind.to(device))
    
            loss = 0.8 * ce_loss + 0.2 * ss_loss
    
            ce_losses.append(ce_loss.item())
            ss_losses.append(ss_loss.item())
            train_losses.append(loss.item())

    return ce_losses, ss_losses, train_losses

# evaluate on full dataset that consists of both known and unknown classes
def evaluate(G, C, dataloader, threshold, KNOWN_CLASSES, UNK_INDEX, device, vis=False):
    C.eval()
    G.eval()
    acc = 0.0
    acc2 = 0.0
    auroc_preds, auroc_preds_2d, auroc_targets = [], [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            out = C(concat_x)[0]

            max_act, indices = torch.max(out, dim=-1)
            
            labels = torch.where(max_act < threshold, UNK_INDEX, indices)
            unk_y = torch.where(torch.isin(y, torch.Tensor(KNOWN_CLASSES).to(device)), y, UNK_INDEX)
            acc += (labels == unk_y).sum().item()

            labels2 = torch.where(max_act < threshold, 1, 0)
            unk_y2 = torch.where(torch.isin(y, torch.Tensor(KNOWN_CLASSES).to(device)), 0, 1)
            acc2 += (labels2 == unk_y2).sum().item()

            # The implicit K+1th class (the open set class) is computed
            # by assuming an extra linear output with constant value 0
            # https://github.com/lwneal/counterfactual-open-set/blob/34fbc726fb7fe76d15fb323e9597c76292b66d81/generativeopenset/evaluation.py#L217
            z = torch.exp(out).sum(dim=1)
            prob_known = z / (z + 1)
            prob_unknown = 1 - prob_known
            
    return  



def main():

    configs = {
        "batch_size": 256,
        "lr":1e-4,
        "betas":(0.5, 0.999),
        "epochs": 10,
        "split": 3,
        "unk_index": 11,
        "ckpt_period":25,
        "generator":"vanilla_ae",
        "classifier":"LeNet",
        "type": "train open-set-classifier"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join("ckpt/openset_ae_mnist_omni", "Mnist_omni")
    out_path = os.path.join("output/openset_ae_mnist_omni", "Mnist_omni")
    

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

 
    test_transform = T.Compose([T.Resize(32),T.Grayscale(num_output_channels=3),T.ToTensor(), #T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    data_man = Mnist_omni_loader(bs=configs["batch_size"],transform=test_transform)
    train_loader = data_man.load_train()
    val_loader = data_man.load_mnist_val()

    generator = torch.load("ckpt/ae_mnist_omni/Mnist_omni/ckpt.pth",weights_only=False).to(device)
    classifier = LeNet_GFROR(num_classes=10,num_transforms=8).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=configs["lr"], betas=configs["betas"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, threshold_mode='abs')

    # freeze generator
    for param in generator.parameters():
        param.requires_grad = False

    # 8 distinct outputs
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


    for i in range(configs["epochs"]):
        train_ce_loss, train_ss_loss, train_loss = train(generator, classifier, train_loader, optimizer, ce_loss, transformations, device)
        val_ce_loss,val_ss_loss,val_loss = evaluate_closedSet(generator, classifier, val_loader, optimizer, ce_loss, transformations, device)
        # test_acc, test_auroc = evaluate(generator, classifier, test_loader, threshold, KNOWN_SPLITS[config.split], config.unk_index, device)
        #scheduler.step(val_ce)

        print('epoch [{}/{}], lr:{:.4f}, train loss:{:.4f}, val_loss: {:.4f}'.format(i+1, configs["epochs"], optimizer.param_groups[0]['lr'], sum(train_loss)/len(train_loss), sum(val_loss)/len(val_loss)))
        
        
    save_path = os.path.join(ckpt_path, "LeNet.pth")
    torch.save(classifier, save_path)

if __name__ == "__main__":
    main()
