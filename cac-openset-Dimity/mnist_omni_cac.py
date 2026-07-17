"""
	Evaluate average performance for our proposed CAC open-set classifier on a given dataset.

	Dimity Miller, 2020
"""


import argparse
import json

import torchvision
import torchvision.transforms as tf
import torch
import torch.nn as nn

from networks import openSetClassifier
import datasets.utils as dataHelper
from utils import find_anchor_means,gather_outputs,Matriz_confusao_osr_dataset_outlier as mc
from sklearn.metrics import accuracy_score

import metrics
import scipy.stats as st
import numpy as np

from Modelos import LeNet_cac
from Datasets import Mnist_omni_loader
from Utils import NOMES,eval_cac,metricasImplementadas
import pandas as pd

import os
device = "cuda:0"
num_classes = 10
epsilons = np.arange(0,1,0.01)

output_dir = "/home/alexandreselani/Desktop/cac-openset-Dimity/Results/Mnist_omni/LeNet"
os.makedirs(output_dir, exist_ok=True)

#loading model
net_weights_dir = "/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet_cac/LeNet_mnist_omni_cac.pt"
net = LeNet_cac(num_classes= num_classes).to(device)
net.load_state_dict(torch.load(net_weights_dir))

#loading training data to adjust anchors
data_manager = Mnist_omni_loader(256, NOMES.LENET_MNIST_OMNI_TRANSFORMS.value)
trainloader = data_manager.load_train()


#find mean anchors for each class
anchor_means = find_anchor_means(net,trainloader,device,num_classes)
net.set_anchors(torch.Tensor(anchor_means))

#obtaining val data distances from anchors
valloader = data_manager.load_gridsearch()
logits,distances,targets = gather_outputs(net,valloader,device)

melhores_hiperparametros = {'epsilon': None}
melhor_f1 = -1
results_all = []

# Mapeia todas as classes presentes na validação para a matriz de confusão
all_targets = np.array(targets)

print("Iniciando Grid Search para o Epsilon...")

for epsilon in epsilons:
    epsilon = round(epsilon, 3)
    print(f"Testando Epsilon: {epsilon}")
    
    predicts, min_scores, scores = net.predict_by_distance(epsilon, distances)
    
    # Conversão explícita para numpy caso retorne tensores do PyTorch
    if isinstance(predicts, torch.Tensor): predicts = predicts.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()
    if isinstance(min_scores, torch.Tensor): min_scores = min_scores.detach().cpu().numpy()
    
    metricas = metricasImplementadas(predicts, targets, -min_scores, metodo="opengan")
    results = metricas._metricas()
    
    # Armazenando os resultados da iteração atual
    res_dict = {
        "epsilon": epsilon,
        "f1_macro": results["F1 macro"],
        "accuracy": results["accuracy"][0] ,
        "uuc_accuracy": results["UUC Accuracy"][0] ,
        "inner_metric": results["inner metric"][0] ,
        "outer_metric": results["outer metric"][0] ,
        "halfpoint": results["halfpoint"][0] ,
        "auroc": results["auroc"]
    }
    results_all.append(res_dict)
    
    # Geração da pasta estruturada por epsilon
    pasta = f"epsilon_{epsilon}/"
    organized_dir = os.path.join(output_dir, pasta)
    os.makedirs(organized_dir, exist_ok=True)
    
    # Salva a matriz de confusão desta configuração
    matriz = mc(predicts, targets, all_targets, [], ["Omniglot", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    matriz.computa_matriz()
    matriz.exibe_matriz(dir=os.path.join(organized_dir,f"matriz_eps_{epsilon}"))
    
    # Atualiza o melhor modelo baseado no F1 Macro
    if results["F1 macro"] > melhor_f1:
        melhor_f1 = results["F1 macro"]
        melhores_hiperparametros["epsilon"] = epsilon

# Salva o arquivo CSV contendo todo o Grid Search
df_grid = pd.DataFrame(results_all)
csv_grid_path = os.path.join(output_dir, "Resultados_grid_cac_lenet.csv")
df_grid.to_csv(csv_grid_path, index=False, float_format="%.3f")
print(f"Grid Search finalizado. Arquivo geral salvo em: {csv_grid_path}")

# ==========================================
# TESTE FINAL COM O MELHOR EPSILON ENCONTRADO
# ==========================================
melhor_epsilon = melhores_hiperparametros["epsilon"]
print(f"\nExecutando teste final com o melhor Epsilon encontrado: {melhor_epsilon}")

testloader = data_manager.load_test()
logits_test, distances_test, targets_test = gather_outputs(net, testloader, device)

predicts_test, min_scores_test, scores_test = net.predict_by_distance(melhor_epsilon, distances_test)

if isinstance(predicts_test, torch.Tensor): predicts_test = predicts_test.detach().cpu().numpy()
if isinstance(targets_test, torch.Tensor): targets_test = targets_test.detach().cpu().numpy()
if isinstance(min_scores_test, torch.Tensor): min_scores_test = min_scores_test.detach().cpu().numpy()

metricas_teste = metricasImplementadas(predicts_test, targets_test, -min_scores_test, metodo="opengan")
results_final = metricas_teste._metricas()

all_targets_test = np.array(targets_test)

# Estruturando dataframe de teste
final_data = [{
    "epsilon": melhor_epsilon,
    "f1_macro": results_final["F1 macro"],
    "accuracy": results_final["accuracy"][0] ,
    "uuc_accuracy": results_final["UUC Accuracy"][0] ,
    "inner_metric": results_final["inner metric"][0] ,
    "outer_metric": results_final["outer metric"][0] ,
    "halfpoint": results_final["halfpoint"][0] ,
    "auroc": results_final["auroc"]
}]

df_final = pd.DataFrame(final_data)

# Gerando a matriz de teste final
matriz_final = mc(predicts_test, targets_test, all_targets_test, [], ["Unknown", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
matriz_final.computa_matriz()
matriz_final.exibe_matriz(dir=os.path.join(output_dir,"melhor_modelo_cac.png"))

filename_final_csv = f"melhor_modelo_teste_eps_{melhor_epsilon}.csv"
final_csv_path = os.path.join(output_dir, filename_final_csv)
df_final.to_csv(final_csv_path, index=False, float_format="%.3f")

print(f"Arquivo de teste final salvo: {final_csv_path}")
print(f"Melhores hiperparâmetros selecionados: {melhores_hiperparametros}")
	





	