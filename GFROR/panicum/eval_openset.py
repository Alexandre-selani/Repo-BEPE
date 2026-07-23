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

from Modelos import LeNet_GFROR
from Datasets import Mnist_omni_loader
from Utils import NOMES, fix_random_seed, metricasImplementadas

fix_random_seed(42)
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=128
MODEL = "ResNet18"
result_dir = f"/home/alexandreselani/Desktop/GFROR/results/mnist_omni/{MODEL}"
generator_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/ae_mnist_omni/Mnist_omni/ckpt.pth"
classifier_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_mnist_omni/Mnist_omni/{MODEL}.pth"
os.makedirs(result_dir,exist_ok=True)


G = torch.load(generator_path,weights_only=False).to(DEVICE)
C = torch.load(classifier_path,weights_only=False).to(DEVICE)

test_transform = T.Compose([T.Resize(32),T.Grayscale(num_output_channels=3),T.ToTensor(), #T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
data_manager = Mnist_omni_loader(BATCH_SIZE,test_transform)

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

def predict(dataloader):
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
    val_loader = data_manager.load_gridsearch()
    results_by_epsilon = {}
    
    max_act, preds, known_score, unknown_score, labels = predict(val_loader)
   
    for epsilon in epsilons:
        predicts = threshold(max_act,preds,epsilon)

        metricas = metricasImplementadas(predicts,labels,outlier_scores=-unknown_score,metodo="opengan")
        metricas = metricas._metricas()
        results_by_epsilon[epsilon] = {
                "epsilon": epsilon,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]}
    
    final_data = []

    for epsilon in sorted(results_by_epsilon.keys()):
        metrics = results_by_epsilon[epsilon]
        final_data.append(metrics)

    df = pandas.DataFrame(final_data)

    os.makedirs(name=result_dir,exist_ok=True)
    df.to_csv(os.path.join(result_dir,"Resultados_model_selection.csv"),index=False,float_format="%.3f")

def test(epsilons):
    test_loader = data_manager.load_test()
    results_by_epsilon = {}
    
    max_act, preds, known_score, unknown_score, labels = predict(test_loader)
    
    for epsilon in epsilons:
        predicts = threshold(max_act,preds,epsilon)

        metricas = metricasImplementadas(predicts,labels,outlier_scores=-unknown_score,metodo="opengan")
        metricas = metricas._metricas()
        results_by_epsilon[epsilon] = {
                "epsilon": epsilon,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]}
    
    final_data = []

    for epsilon in sorted(results_by_epsilon.keys()):
        metrics = results_by_epsilon[epsilon]
        final_data.append(metrics)

    df = pandas.DataFrame(final_data)

    os.makedirs(name=result_dir,exist_ok=True)
    df.to_csv(os.path.join(result_dir,"Resultados_test.csv"),index=False,float_format="%.3f")
    

thresholds = np.arange(0,30,0.5)
#train()

val(thresholds)
test(thresholds)
