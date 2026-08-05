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
from Modelos import LeNetFeaturizer,ResNet18Featurizer
from Datasets import Mnist_omni_loader
from Utils import metricasImplementadasV2,NOMES,metricLogger
import pandas
import os
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=128
MODEL = "LeNet"
METHOD = "COSTARR"
DATASET = "MNIST/OMNI"
MC_NAME = f"{METHOD} - {DATASET}"
result_dir = f"/home/alexandreselani/Desktop/COSTARR/results/mnist_omni/{MODEL}"
calcs_dir = f"/home/alexandreselani/Desktop/COSTARR/calcs/mnist_omni/{MODEL}/"
os.makedirs(result_dir,exist_ok=True)
os.makedirs(calcs_dir,exist_ok=True)  

final_calcs_dir = os.path.join(calcs_dir,"mnist_costarr.pt")
# model = LeNetFeaturizer().to(DEVICE)
# 

model = None
transform = None
if MODEL == "LeNet":
    model = LeNetFeaturizer().to(DEVICE)
    transform = NOMES.LENET_MNIST_OMNI_TRANSFORMS.value
elif MODEL == "ResNet18":
    model = ResNet18Featurizer().to(DEVICE)
    transform = NOMES.RESNET18_MNIST_OMNI_TRANSFORMS.value

model.eval()
model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_mnist_omni/{MODEL}/{MODEL}_mnist_omni.pt"))


data_manager = Mnist_omni_loader(BATCH_SIZE,transform)
costarr_calcs=None

def train():
    train_loader = data_manager.load_train()
    costarrFit(model,train_loader,final_calcs_dir)
    


def val(epsilons):
    val_loader = data_manager.load_gridsearch()
    results_by_epsilon = {}
    
    scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)
    metric_logger = metricLogger(epsilons,0,os.path.join(result_dir,"Val"),mc_column_names=["Omniglot",0,1,2,3,4,5,6,7,8,9],mc_title=MC_NAME)
    for epsilon in epsilons:
        predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

        metricas = metricasImplementadasV2(predicts,labels,outlier_scores=scores,metodo="opengan")
        metricas = metricas._metricas()
        metric_logger.update(metrics=metricas,fold=0,epsilon=epsilon)
        metric_logger.update_mc(epsilon,predicts,labels,labels)
    metric_logger.aggregate("Model_selection.csv")

def test(epsilons):
    val_loader = data_manager.load_test()
    results_by_epsilon = {}
    
    scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)
    metric_logger = metricLogger(epsilons,0,os.path.join(result_dir,"Test"),mc_column_names=["Omniglot",0,1,2,3,4,5,6,7,8,9],mc_title=MC_NAME)
    for epsilon in epsilons:
        predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

        metricas = metricasImplementadasV2(predicts,labels,outlier_scores=scores,metodo="opengan")
        metricas = metricas._metricas()
        metric_logger.update(metrics=metricas,fold=0,epsilon=epsilon)
        metric_logger.update_mc(epsilon,predicts,labels,labels)
    metric_logger.aggregate("Test.csv")

thresholds = np.arange(0.2,0.7,0.01)
#train()

if not costarr_calcs:
    costarr_calcs = torch.load(final_calcs_dir,weights_only=False)
val(thresholds)
test(thresholds)
