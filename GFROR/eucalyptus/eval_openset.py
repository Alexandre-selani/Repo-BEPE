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
from torchvision.models import AlexNet_Weights
from collections import Counter
from torchvision.utils import make_grid, save_image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, auc
from torchmetrics import AUROC

from model.vanilla_ae import VanillaAE
from model.wgan import WGAN_GP
from model.utils import to_img, to_4d

from Modelos import ResNet18_GFROR
from Datasets import Eucalyptus_openset_loader
from Utils import NOMES, fix_random_seed, metricasImplementadasV2,metricLogger

fix_random_seed(42)
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=20
MODEL = "AlexNet"
N_FOLDS = 5

result_dir = f"/home/alexandreselani/Desktop/GFROR/results/eucalyptus/{MODEL}"
generator_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/ae_eucalyptus/eucalyptus/"
classifier_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_eucalyptus/{MODEL}"
os.makedirs(result_dir,exist_ok=True)

weights = AlexNet_Weights.IMAGENET1K_V1

test_transform = weights.transforms()
data_manager = Eucalyptus_openset_loader(BATCH_SIZE)

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

def predict(dataloader,G,C):
    G.eval()
    C.eval()

    all_targets = []
    all_max_act = []
    all_idx = []
    all_prob_known = []
    all_prob_unknown = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            out = C(concat_x)[0]

            max_act, indices = torch.max(out, dim=-1)

            z = torch.exp(out).sum(dim=1)
            prob_known = z / (z + 1)
            prob_unknown = 1 - prob_known

            all_targets.append(y.cpu())
            all_max_act.append(max_act.cpu())
            all_idx.append(indices.cpu())
            all_prob_known.append(prob_known.cpu())
            all_prob_unknown.append(prob_unknown.cpu())

    all_targets = torch.cat(all_targets, dim=0)
    all_max_act = torch.cat(all_max_act, dim=0)
    all_idx = torch.cat(all_idx, dim=0)
    all_prob_known = torch.cat(all_prob_known, dim=0)
    all_prob_unknown = torch.cat(all_prob_unknown, dim=0)

    print(all_prob_known,all_prob_unknown)
    return all_max_act, all_idx, all_prob_known, all_prob_unknown, all_targets

def threshold(max_act,preds,epsilon):
    print(pandas.DataFrame(max_act).describe(),preds)
    predict = torch.where(max_act < epsilon, -1, preds)
    return predict

def val(epsilons):

    logger = metricLogger(epsilons,N_FOLDS,os.path.join("/home/alexandreselani/Desktop/GFROR/results/eucalyptus/Val/"))
    for fold in range(N_FOLDS):
        val_loader = data_manager.load_val(fold,test_transform)
        
        
        G = torch.load(os.path.join(generator_path,f"fold_{fold}.pth"),weights_only=False).to(DEVICE)
        C = torch.load(os.path.join(classifier_path,f"Fold_{fold}","ckpt.pth"),weights_only=False).to(DEVICE)

        max_act, preds, known_score, unknown_score, labels = predict(val_loader,G,C)

        for epsilon in epsilons:

            predicts = threshold(max_act,preds,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=-unknown_score,metodo="opengan")
            metricas = metricas._metricas()

            logger.update(metricas,fold,epsilon)
            logger.update_mc(epsilon,predicts,labels,labels)
    logger.aggregate("Val.csv")

def test(epsilons):
    logger = metricLogger(epsilons,N_FOLDS,os.path.join("/home/alexandreselani/Desktop/GFROR/results/eucalyptus/Test/"))

    for fold in range(N_FOLDS):
        test_loader = data_manager.load_test(fold,test_transform)
        
        
        G = torch.load(os.path.join(generator_path,f"fold_{fold}.pth"),weights_only=False).to(DEVICE)
        C = torch.load(os.path.join(classifier_path,f"Fold_{fold}","ckpt.pth"),weights_only=False).to(DEVICE)

        max_act, preds, known_score, unknown_score, labels = predict(test_loader,G,C)

        for epsilon in epsilons:

            predicts = threshold(max_act,preds,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=-unknown_score,metodo="opengan")
            metricas = metricas._metricas()

            logger.update(metricas,fold,epsilon)
            logger.update_mc(epsilon,predicts,labels,labels)
    logger.aggregate("Test.csv")
    

thresholds = np.arange(-1,5,0.1)
#train()

val(thresholds)
test(thresholds)
