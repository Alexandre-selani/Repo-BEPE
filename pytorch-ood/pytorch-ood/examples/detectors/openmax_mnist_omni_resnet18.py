"""
OpenMax
==============================

:class:`OpenMax <pytorch_ood.detector.OpenMax>` was originally proposed
for Open Set Recognition but can be adapted for Out-of-Distribution tasks.

.. warning:: OpenMax requires ``libmr`` to be installed, which is broken at the moment. You can only use it
   by installing ``cython`` and ``numpy``, and ``libmr`` manually afterwards.


"""

"""label = -1 --> desconhecido
label >= 0 --> conhecido

O primeiro item dos logits é referente ao score para classe desconhecida, o restante é para as classes conhecidas

TODO: 
    REDE NEURAL MINHA: feito 
    LOOP DE TREINAMENTO E TESTES: feito
    ANALISE GRAFICA? feito
"""
import sys
sys.path = [p for p in sys.path if "BRACIS" not in p]
from pytorch_ood.utils import Matriz_confusao_osr_dataset_outlier_cumulativa as mc
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.detector import OpenMax
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc
import copy
import math
from Utils import fix_random_seed,NOMES,metricasImplementadasV2,metricLogger
from Utils.Model_utils import train,eval
from Datasets import Mnist_omni_loader
from Modelos import ResNet18
seed = 42
fix_random_seed(seed)

device = "cuda:0"

num_classes = 10
# Colunas da matriz de confusao: a classe desconhecida primeiro (predict_unknown_value=0),
# depois os digitos do MNIST em ordem.
column_names = ["Unknown"] + [str(i) for i in range(num_classes)]

def test(test_loader,detector):
    predicts=[]
    labels=[]
    outlier_scores = []

    detector.model.eval()
    for X, y in test_loader:
        
        #score eh a ativacao de todas as classes apos a openmax
        with torch.no_grad():
            score = detector(X.to(device))
            
            #print(score)
            max_values, predicted = torch.max(score, dim=1)
            predict = torch.where(max_values >= detector.epsilon, predicted, torch.zeros_like(predicted))

        outlier_scores.append(score[:, 0].detach().cpu())
        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
        
    outlier_scores = torch.cat(outlier_scores, dim=0).cpu().numpy()
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    
    #print(labels.shape,outlier_scores.shape)
    #ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadasV2(predict=predicts, label=labels, outlier_scores=outlier_scores,metodo="openmax")

    #print(ood_metrics.compute())
    #print(predicts,labels)
    
    return metricas._metricas(),predicts,labels  


min_tailsize = 0
max_tailsize = 1000
step_tail = 100
tails = list(range(min_tailsize, max_tailsize+1, step_tail))
alphas = [2,4,6,8,10]
epsilons = np.arange(0.2,1,0.2)

def grid_search():
    nomeDataset = "Mnist_omni"
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_mnist_omniglot/Resultados OpenMax/ResNet18/Val/"
    os.makedirs(output_dir, exist_ok=True)
    
    
    data_manager = Mnist_omni_loader(bs=256,transform=NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value)

    # Melhor combinacao vista ate agora, pelo F1 macro.
    best_f1 = -1.0
    best_params = None
    best_metricas = None

    for alpha in alphas:
        for epsilon in epsilons:
            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            organized_dir = os.path.join(output_dir, pasta)
            os.makedirs(organized_dir, exist_ok=True)

            registra_metricas = metricLogger(tails,0,organized_dir,mc_column_names=column_names,predict_unknown_value=0)

            
            gc.collect()
            torch.cuda.empty_cache()

            model = ResNet18(num_classes=10)
            model.load_state_dict(torch.load("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"))
            model.to(device=device)
            model.eval()

            train_dataloader, val_dataloader = data_manager.load_train(), data_manager.load_gridsearch()

            all_targets = np.array([])
            for (_, y) in val_dataloader:
                all_targets = np.append(all_targets, y.detach().cpu().numpy())


            for tail in tails:
                print(alpha,epsilon,tail)
                gc.collect()
                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                detector.fit(train_dataloader, device=device)

                metricas, predicts, targets_val = test(val_dataloader, detector)

                registra_metricas.update(metricas,0,tail)
                registra_metricas.update_mc(tail,predicts,targets_val,all_targets)

                if metricas["F1 macro"] > best_f1:
                    best_f1 = metricas["F1 macro"]
                    best_params = {"alpha": alpha, "epsilon": epsilon, "tail": tail}
                    best_metricas = metricas

                del detector

            registra_metricas.aggregate("gridsearch.csv")

    print("\n" + "=" * 60)
    print("MELHOR COMBINACAO NO GRID SEARCH (por F1 macro)")
    print("=" * 60)
    if best_params is None:
        print("Nenhuma combinacao avaliada.")
    else:
        print("alpha   = {}".format(best_params["alpha"]))
        print("epsilon = {}".format(best_params["epsilon"]))
        print("tail    = {}".format(best_params["tail"]))
        print("-" * 60)
        for chave, valor in best_metricas.items():
            print("{:<16}: {:.4f}".format(chave, valor))
    print("=" * 60)

def test_hiperparameters():
    nomeDataset = "Mnist_omni"
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_mnist_omniglot/Resultados OpenMax/ResNet18/Test/"
    os.makedirs(output_dir, exist_ok=True)


    data_manager = Mnist_omni_loader(bs=256,transform=NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value)

    for alpha in alphas:
        for epsilon in epsilons:
            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            organized_dir = os.path.join(output_dir, pasta)
            os.makedirs(organized_dir, exist_ok=True)

            registra_metricas = metricLogger(tails,0,organized_dir,mc_column_names=column_names,predict_unknown_value=0)

            
            gc.collect()
            torch.cuda.empty_cache()

            model = ResNet18(num_classes=10)
            model.load_state_dict(torch.load("/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"))
            model.to(device=device)
            model.eval()

            train_dataloader, test_dataloader = data_manager.load_train(), data_manager.load_test()

            all_targets = np.array([])
            for (_, y) in test_dataloader:
                all_targets = np.append(all_targets, y.detach().cpu().numpy())


            for tail in tails:
        
                gc.collect()
                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                detector.fit(train_dataloader, device=device)

                metricas, predicts, targets_val = test(test_dataloader, detector)

                registra_metricas.update(metricas,0,tail)
                registra_metricas.update_mc(tail,predicts,targets_val,all_targets)

                del detector

            registra_metricas.aggregate("Test.csv")
        

if __name__ == '__main__':
    grid_search()

    test_hiperparameters()
