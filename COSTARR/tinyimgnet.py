import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from funcs import *
from Modelos import ResNet18_tinyimgnet_featurizer
from Datasets import TinyImageNet_loader
from Utils import metricasImplementadasV2,metricLogger,NOMES,Matriz_confusao_osr_dataset_outlier_cumulativa as mc
import pandas as pd

import pandas
import os
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
MODEL = "ResNet18"
N_FOLDS = 5
METHOD = "COSTARR"
DATASET = "Tinyimgnet"
MC_NAME = f"{METHOD} - {DATASET}"

result_dir = f"/home/alexandreselani/Desktop/COSTARR/results/tinyimgnet/{MODEL}"
calcs_dir = f"/home/alexandreselani/Desktop/COSTARR/calcs/tinyimgnet/{MODEL}/"
os.makedirs(result_dir,exist_ok=True)
os.makedirs(calcs_dir,exist_ok=True)  


# model = LeNetFeaturizer().to(DEVICE)
# 

data_manager = TinyImageNet_loader()



def train():

    for fold in range(N_FOLDS):
        train_loader = data_manager.get_train_loader(fold,data_manager.eval_transforms[fold])

        model = ResNet18_tinyimgnet_featurizer(num_classes=20).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/{MODEL}/Split_{fold}/{MODEL}_TinyImageNet_split_{fold}.pt"))

        final_calcs_dir = os.path.join(calcs_dir,f"tinyimgnet_split_{fold}_costarr.pt")

        costarrFit(model,train_loader,final_calcs_dir)
    


def val(epsilons):
    val_dir = os.path.join(result_dir,"Val")
    os.makedirs(val_dir,exist_ok=True)

    metric_logger = metricLogger(epsilons,N_FOLDS,val_dir,mc_title=MC_NAME)
    for fold in range(N_FOLDS):
        
        val_loader = data_manager.get_val_loader(fold,data_manager.eval_transforms[fold])
        costarr_calcs=torch.load(os.path.join(calcs_dir,f"tinyimgnet_split_{fold}_costarr.pt"))

        model = ResNet18_tinyimgnet_featurizer(num_classes=20).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/{MODEL}/Split_{fold}/{MODEL}_TinyImageNet_split_{fold}.pt"))

        scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)
    
        for epsilon in epsilons:
            predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

            metricas = metricasImplementadasV2(predicts,labels,outlier_scores=scores,metodo="opengan")
            metricas = metricas._metricas()
            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)

    metric_logger.aggregate("Val.csv")

            

def test(epsilons):
    test_dir = os.path.join(result_dir,"Test")
    os.makedirs(test_dir,exist_ok=True)

    metric_logger = metricLogger(epsilons,N_FOLDS,test_dir,mc_title=MC_NAME)
    for fold in range(N_FOLDS):
        
        test_loader = data_manager.get_test_loader(fold,data_manager.eval_transforms[fold])
        costarr_calcs=torch.load(os.path.join(calcs_dir,f"tinyimgnet_split_{fold}_costarr.pt"))

        model = ResNet18_tinyimgnet_featurizer(num_classes=20).to(DEVICE)
        model.eval()
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/{MODEL}/Split_{fold}/{MODEL}_TinyImageNet_split_{fold}.pt"))

        scores, max_logits,max_logits_idx, labels = costarrPredict(model,test_loader,costarr_calcs)
    
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


