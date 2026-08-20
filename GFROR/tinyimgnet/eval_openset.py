"""Avaliacao open-set do GFROR no TinyImageNet.

Avalia o classificador produzido por tinyimgnet/train_openset.py, pareado com o
autoencoder de tinyimgnet/train_generator.py: o AE gera x_hat, o classificador
recebe (x, x_hat) em 6 canais e a cabeca de classificacao da a ativacao usada
tanto para o rotulo quanto para o escore de rejeicao.

O pre-processamento e o mesmo do treino (Resize(64) + ToTensor(), sem
normalizacao): o AE foi treinado nessa escala e reconstroi em [0, 1].
"""

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

from model.vanilla_ae import VanillaAE64
from model.utils import to_img, to_4d

from Modelos import ResNet18_tinyimgnet_GFROR
from Datasets import TinyImageNet_loader
from Utils import NOMES, fix_random_seed, metricasImplementadasV2,metricLogger

fix_random_seed(42)
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=128
MODEL = "ResNet18"
N_SPLITS = 5
NUM_CLASSES = 20
IMAGE_SIZE = 64

DATA_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"

result_dir = f"/home/alexandreselani/Desktop/GFROR/results/tinyimgnet/{MODEL}"
generator_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/ae_tinyimgnet/Tinyimgnet/"
classifier_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_tinyimgnet/{MODEL}"
os.makedirs(result_dir,exist_ok=True)

column_names = ["Unknown"] + [str(i) for i in range(NUM_CLASSES)]

test_transform = T.Compose([T.Resize((IMAGE_SIZE,IMAGE_SIZE)),T.ToTensor(), #sem normalizacao: o AE foi treinado em [0,1]
    ])
data_manager = TinyImageNet_loader(
    data_dir=DATA_DIR,
    splits_dir=SPLITS_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
)

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

    val_dir = os.path.join(result_dir,"Val")
    os.makedirs(val_dir,exist_ok=True)

    logger = metricLogger(epsilons,N_SPLITS,val_dir,mc_column_names=column_names)
    for split in range(N_SPLITS):
        val_loader = data_manager.get_val_loader(split,test_transform)


        G = torch.load(os.path.join(generator_path,f"split_{split}.pth"),weights_only=False,map_location=DEVICE).to(DEVICE)
        C = torch.load(os.path.join(classifier_path,f"Split_{split}","ckpt.pth"),weights_only=False,map_location=DEVICE).to(DEVICE)

        max_act, preds, known_score, unknown_score, labels = predict(val_loader,G,C)

        for epsilon in epsilons:

            predicts = threshold(max_act,preds,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=-unknown_score,metodo="opengan")
            metricas = metricas._metricas()

            logger.update(metricas,split,epsilon)
            logger.update_mc(epsilon,predicts,labels,labels)
    logger.aggregate("Val.csv")

def test(epsilons):

    test_dir = os.path.join(result_dir,"Test")
    os.makedirs(test_dir,exist_ok=True)

    logger = metricLogger(epsilons,N_SPLITS,test_dir,mc_column_names=column_names)

    for split in range(N_SPLITS):
        test_loader = data_manager.get_test_loader(split,test_transform)


        G = torch.load(os.path.join(generator_path,f"split_{split}.pth"),weights_only=False,map_location=DEVICE).to(DEVICE)
        C = torch.load(os.path.join(classifier_path,f"Split_{split}","ckpt.pth"),weights_only=False,map_location=DEVICE).to(DEVICE)

        max_act, preds, known_score, unknown_score, labels = predict(test_loader,G,C)

        for epsilon in epsilons:

            predicts = threshold(max_act,preds,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=-unknown_score,metodo="opengan")
            metricas = metricas._metricas()

            logger.update(metricas,split,epsilon)
            logger.update_mc(epsilon,predicts,labels,labels)
    logger.aggregate("Test.csv")


thresholds = np.arange(0,30,0.2)
#train()

#val(thresholds)
test(thresholds)
