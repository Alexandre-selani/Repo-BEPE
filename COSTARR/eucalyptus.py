import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18,AlexNet_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from funcs import *
from Modelos import AlexNetFeaturizer
from Datasets import Eucalyptus_openset_loader
from Utils import metricasImplementadasV2,metricLogger,NOMES,Matriz_confusao_osr_dataset_outlier_cumulativa as mc
import pandas as pd

import pandas
import os
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
MODEL = "AlexNet"
DATASET = "dataset-1"
METHOD = "COSTARR"
MC_TITLE = f"{METHOD} - Eucalyptus"
MC_COLUMN_NAMES = ["Ceratocystis","Ground","Healthy"]
N_FOLDS = 5

# ─── Caminhos base ───────────────────────────────────────────────────────
BASE_DIR = "/home/alexandreselani/Desktop"
BASE_COSTARR = f"{BASE_DIR}/COSTARR"
BASE_EUCALYPTUS_MODELS = f"{BASE_DIR}/Eucalyptus/OpenSet/Models"

result_dir = f"{BASE_COSTARR}/results/eucalyptus/{DATASET}/{MODEL}"
calcs_dir = f"{BASE_COSTARR}/calcs/eucalyptus/{DATASET}/{MODEL}/"
os.makedirs(result_dir,exist_ok=True)
os.makedirs(calcs_dir,exist_ok=True)  

weights = AlexNet_Weights.IMAGENET1K_V1

transforms = weights.transforms()

# model = LeNetFeaturizer().to(DEVICE)
# 

data_manager = Eucalyptus_openset_loader(BATCH_SIZE,DATASET)



def train():

    for fold in range(N_FOLDS):
        train_loader = data_manager.load_train(fold,transforms)

        model = AlexNetFeaturizer(num_classes=2).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(f"{BASE_EUCALYPTUS_MODELS}/{DATASET}/AlexNet_fold_{fold}.pt"))

        final_calcs_dir = os.path.join(calcs_dir,f"eucalyptus_fold_{fold}_costarr.pt")

        costarrFit(model,train_loader,final_calcs_dir)
    


def val(epsilons):
    val_dir = os.path.join(result_dir,"Val")
    os.makedirs(val_dir,exist_ok=True)

    metric_logger = metricLogger(epsilons,N_FOLDS,val_dir,mc_column_names=MC_COLUMN_NAMES,mc_title=MC_TITLE)
    for fold in range(N_FOLDS):
        fold_results = []
        val_loader = data_manager.load_val(fold,transforms)
        costarr_calcs=torch.load(os.path.join(calcs_dir,f"eucalyptus_fold_{fold}_costarr.pt"))

        model = AlexNetFeaturizer(num_classes=2).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(f"{BASE_EUCALYPTUS_MODELS}/{DATASET}/{MODEL}_fold_{fold}.pt"))

        scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)

        
        for epsilon in epsilons:
            predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=scores,metodo="opengan")
            metricas = metricas._metricas()

            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)
    metric_logger.aggregate("Val.csv")
    

def test(epsilons):
    val_dir = os.path.join(result_dir,"Test")
    os.makedirs(val_dir,exist_ok=True)

    metric_logger = metricLogger(epsilons,N_FOLDS,val_dir,mc_column_names=MC_COLUMN_NAMES,mc_title=MC_TITLE)
    for fold in range(N_FOLDS):
        fold_results = []
        val_loader = data_manager.load_test(fold,transforms)
        costarr_calcs=torch.load(os.path.join(calcs_dir,f"eucalyptus_fold_{fold}_costarr.pt"))

        model = AlexNetFeaturizer(num_classes=2).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(f"{BASE_EUCALYPTUS_MODELS}/{DATASET}/{MODEL}_fold_{fold}.pt"))

        scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)

        
        for epsilon in epsilons:
            predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=scores,metodo="opengan")
            metricas = metricas._metricas()

            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)
    metric_logger.aggregate("Test.csv")
    

thresholds = np.arange(0,1,0.01)
#train()
val(thresholds)
test(thresholds)


