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
from Utils import metricasImplementadas,NOMES
import pandas
import os
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=128
MODEL = "LeNet"
result_dir = f"/home/alexandreselani/Desktop/COSTARR/results/mnist_omni/{MODEL}"
calcs_dir = f"/home/alexandreselani/Desktop/COSTARR/calcs/mnist_omni/{MODEL}/"
os.makedirs(result_dir,exist_ok=True)
os.makedirs(calcs_dir,exist_ok=True)  

final_calcs_dir = os.path.join(calcs_dir,"mnist_costarr.pt")
# model = LeNetFeaturizer().to(DEVICE)
# 

model = LeNetFeaturizer().to(DEVICE)
model.eval()
model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_mnist_omni/{MODEL}/{MODEL}_mnist_omni.pt"))

data_manager = Mnist_omni_loader(BATCH_SIZE,NOMES.LENET_MNIST_OMNI_TRANSFORMS.value)
costarr_calcs=None

def train():
    train_loader = data_manager.load_train()
    costarrFit(model,train_loader,final_calcs_dir)
    


def val(epsilons):
    val_loader = data_manager.load_gridsearch()
    results_by_epsilon = {}
    
    scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)
   
    for epsilon in epsilons:
        predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

        metricas = metricasImplementadas(predicts,labels,outlier_scores=scores,metodo="opengan")
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
    val_loader = data_manager.load_test()
    results_by_epsilon = {}
    
    scores, max_logits,max_logits_idx, labels = costarrPredict(model,val_loader,costarr_calcs)

    for epsilon in epsilons:
        predicts = thresholdPredicitions(scores,max_logits_idx,epsilon)

        metricas = metricasImplementadas(predicts,labels,outlier_scores=scores,metodo="opengan")
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

thresholds = np.arange(0,1,0.01)
#train()

if not costarr_calcs:
    costarr_calcs = torch.load(final_calcs_dir,weights_only=False)
val(thresholds)
test(thresholds)
