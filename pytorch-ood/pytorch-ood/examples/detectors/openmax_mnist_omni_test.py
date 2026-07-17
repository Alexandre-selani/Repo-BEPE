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
"""
from torch.utils.data import DataLoader
import torch
from torchvision.models import alexnet
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.detector import OpenMax
from pytorch_ood.utils import Matriz_confusao_osr_dataset_outlier_cumulativa as mc
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc
from Modelos.ResNet18_backbone import ResNet18
from Utils import fix_random_seed,metricasImplementadas,NOMES
from Datasets.Load_Data_mnist_omni import Mnist_omni_loader

seed = 42
fix_random_seed(seed)

device = "cuda:0"



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
    
    print(labels.shape,outlier_scores.shape)
    #ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadas(predict=predicts, label=labels, outlier_scores=outlier_scores,metodo="openmax")

    #print(ood_metrics.compute())
    #print(predicts,labels)
    
    return metricas._metricas(),predicts,labels  

def confusion_matrix(test_loader,targets_original,nome_classes_originais,UUC_classes,detector):
    predicts=[]
    labels=[]

    
    for X, y in test_loader:
        #score eh a ativacao de todas as classes apos a openmax
        with torch.no_grad():
            score = detector(X.to(device))
            
            max_values, predicted = torch.max(score, dim=1)
            predict = torch.where(max_values >= detector.epsilon, predicted, torch.zeros_like(predicted))

        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
    
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()

    matriz_confusao = mc(predicts,labels,targets_original,UUC_classes,nome_classes_originais)
    matriz_confusao.computa_matriz()
    matriz_confusao.exibe_matriz()


def main():
    
    # Diretório de saída
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_mnist_omniglot/Resultados OpenMax/"
    os.makedirs(output_dir, exist_ok=True)
    
    model = ResNet18(num_classes=10)
    model.load_state_dict(torch.load(NOMES.RESNET18_MNIST_OMNI.value))
    model.to(device=device)
    bs = 256


    data = Mnist_omni_loader(bs)

    
    test_dataloader = data.load_test()
    train_dataloader = data.load_train()
    
    tail = 600
    alpha = 3
    epsilon = 0.9

    detector = OpenMax(model, tailsize=tail,
                              alpha=alpha, euclid_weight=1, 
                              epsilon=epsilon)       
    
    # Ajuste (Fit) - Geralmente precisa passar os dados de treino para calcular os centros/weibulls
    detector.fit(train_dataloader, device=device)
    
    metricas,predicts,targets_test = test(test_dataloader, detector)

            
    all_targets = np.array([])

    for (_, y) in test_dataloader:
        for target in y:
            all_targets= np.append(all_targets,target.detach().cpu())
    
    results = {
                "tail": tail,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]
                }
    
    

    df = pd.DataFrame([results])

    matriz = mc(predicts,targets_test,all_targets,[],["Omniglot",0,1,2,3,4,5,6,7,8,9])
    matriz.computa_matriz()

    
    filename_csv = f"test_eps_{epsilon}_alpha_{alpha}_tail_{tail}.csv"
    organized_dir = os.path.join(output_dir, filename_csv)
    
    matriz.exibe_matriz(dir=output_dir,name=f"test.png")

    df.to_csv(organized_dir, index=False,float_format="%.3f")
    print(f"Arquivo salvo: melhor resultado")
   
    
if __name__ == '__main__':
    main()



