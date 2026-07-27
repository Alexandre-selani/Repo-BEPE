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
from Datasets import Panicum_halfsize_loader
from Modelos import ResNet18
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
    
    #print(labels.shape,outlier_scores.shape)
    #ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadasV2(predict=predicts, label=labels, outlier_scores=outlier_scores,metodo="openmax")

    #print(ood_metrics.compute())
    #print(predicts,labels)
    
    return metricas._metricas(),predicts,labels  


def grid_search():
    nomeDataset = "Panicum"
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/teste_metricLogger/Val/"
    os.makedirs(output_dir, exist_ok=True)
    
    
    data_manager = Panicum_halfsize_loader(bs=32)

    min_tailsize = 0
    max_tailsize = 30
    step_tail = 5
    tails = list(range(min_tailsize, max_tailsize+1, step_tail))
    alphas = [2]
    epsilons = np.arange(0,1,0.2)

    

    for alpha in alphas:
        for epsilon in epsilons:

            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            organized_dir = os.path.join(output_dir, pasta)
            os.makedirs(organized_dir, exist_ok=True)

            registra_metricas = metricLogger(tails,5,organized_dir)

            for fold in range(5):
                gc.collect()
                torch.cuda.empty_cache()

                model = ResNet18(num_classes=2)
                model.load_state_dict(torch.load(os.path.join(NOMES.PANICUM_RESNET.value, f"Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt")))
                model.to(device=device)
                model.eval()

                train_dataloader, val_dataloader = data_manager.load_train(fold,NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value), data_manager.load_val(fold,NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value)

                all_targets = np.array([])
                for (_, y) in val_dataloader:
                    all_targets = np.append(all_targets, y.detach().cpu().numpy())


                for tail in tails:
                    print(f"Fold:{fold} tail {tail} alpha {alpha}, epsilon {epsilon}")
                    gc.collect()
                    detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                    detector.fit(train_dataloader, device=device)

                    metricas, predicts, targets_val = test(val_dataloader, detector)

                    registra_metricas.update(metricas,fold,tail)
                    registra_metricas.update_mc(tail,predicts,targets_val,all_targets)

                    del detector
                
            registra_metricas.aggregate("gridsearch.csv")

def test_hiperparameters(alphas, epsilons, tails):
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum_plantnet/Test/"
    os.makedirs(output_dir, exist_ok=True)
    
    data_manager = Panicum_halfsize_loader(bs=32)
    

    for alpha in alphas:
        for epsilon in epsilons:
            
            # 1. Cria um dicionário para segurar as linhas de cada fold
            fold_data_dict = {fold: [] for fold in range(5)}
            final_data = []
            for tail in tails:
                results_by_tail = {tail: {'f1': [], 'acc': [], 'uuc_acc': [], 'inner': [], 'outer': [], 'half': [], 'auroc': []}}
                matrizes_confusao_acumulada = {tail: None}

                for fold in range(5):
                    print(f"Testando Alpha: {alpha}, Epsilon: {epsilon}, Tail: {tail} | Fold {fold}...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    model = ResNet18(num_classes=2)
                    model.load_state_dict(torch.load(os.path.join(NOMES.PANICUM_RESNET.value, f"Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt")))
                    model.to(device=device)
                    model.eval()
                    
                    train_dataloader, test_dataloader = data_manager.load_train(fold, NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value), data_manager.load_test(fold, NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value)
                    
                    all_targets = np.array([])
                    for (_, y) in test_dataloader:
                        all_targets = np.append(all_targets, y.detach().cpu().numpy())
                        
                    detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                    detector.fit(train_dataloader, device=device)
                    
                    metricas, predicts, targets_test = test(test_dataloader, detector)
                    
                    # Acumula para as médias (código original)
                    results_by_tail[tail]['f1'].append(metricas["F1 macro"])
                    results_by_tail[tail]['acc'].append(metricas["accuracy"][0])
                    results_by_tail[tail]['uuc_acc'].append(metricas["UUC Accuracy"][0])
                    results_by_tail[tail]['inner'].append(metricas["inner metric"][0])
                    results_by_tail[tail]['outer'].append(metricas["outer metric"][0])
                    results_by_tail[tail]['half'].append(metricas["halfpoint"][0])
                    results_by_tail[tail]['auroc'].append(metricas['auroc'])

                    # 2. Salva o resultado ESPECÍFICO DESTE FOLD
                    fold_data_dict[fold].append({
                        "tail": tail,
                        "f1": metricas["F1 macro"],
                        "acc": metricas["accuracy"][0],
                    #    "uuc_acc": metricas["UUC Accuracy"][0],
                        "inner": metricas["inner metric"][0],
                        "outer": metricas["outer metric"][0],
                        "half": metricas["halfpoint"][0],
                        "auroc": metricas["auroc"]
                    })

                    if matrizes_confusao_acumulada[tail] is None:
                        matriz = mc(predicts, targets_test, all_targets, [], ["Panicum", "Ground", "Healthy"])
                        matriz.computa_matriz()
                        matrizes_confusao_acumulada[tail] = matriz
                    else:
                        matrizes_confusao_acumulada[tail].set_data(predicts, targets_test, all_targets)
                        matrizes_confusao_acumulada[tail].computa_matriz()

                    del detector, model

                
                # Processa médias após os 5 folds
                metrics = results_by_tail[tail]
                row = {
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "tail": tail,
                    "f1_macro_mean": np.mean(metrics['f1']),
                    "f1_macro_std": np.std(metrics['f1']),
                    "acc_mean": np.mean(metrics['acc']),
                    "acc_std": np.std(metrics['acc']),
                  #  "uuc_acc_mean": np.mean(metrics['uuc_acc']),
                  #  "uuc_acc_std": np.std(metrics['uuc_acc']),
                    "inner_mean": np.mean(metrics['inner']),
                    "inner std": np.std(metrics["inner"]),
                    "outer_mean": np.mean(metrics['outer']),
                    "outer_std": np.std(metrics["outer"]),
                    "halfpoint_mean": np.mean(metrics['half']),
                    "halfpoint_std": np.std(metrics['half']),
                    "auroc_mean": np.mean(metrics['auroc']),
                    "auroc_std": np.std(metrics['auroc'])
                }
                final_data.append(row)
                
                matrizes_confusao_acumulada[tail].exibe_matriz(dir=output_dir, name=f"alpha_{alpha}_tail_{tail}_eps_{epsilon}_final")

                    # CSV Geral Agregado
            df = pd.DataFrame(final_data)
            filename_csv = f"Results_test_alpha_{alpha}_eps_{epsilon}.csv"
            csv_path = os.path.join(output_dir, filename_csv)
            df.to_csv(csv_path, index=False, float_format="%.3f")
            print(f"Processo finalizado. Arquivo geral salvo em: {csv_path}")

            # ==========================================
            # 3. NOVO: Salva os arquivos de cada fold para o atual Alpha/Epsilon
            # ==========================================
            for fold in range(5):
                df_fold = pd.DataFrame(fold_data_dict[fold])
                nome_arquivo_fold = f"Results_Test_Fold_{fold}_alpha_{alpha}_eps_{epsilon}.csv"
                caminho_fold = os.path.join(output_dir,f"alpha_{alpha}",f"epsilon_{epsilon}","Folds")
                os.makedirs(caminho_fold,exist_ok=True)
                caminho_fold = os.path.join(caminho_fold,nome_arquivo_fold)
                df_fold.to_csv(caminho_fold, index=False, float_format="%.3f")
                print(f"[*] Arquivo do Fold {fold} gerado: {nome_arquivo_fold}")

        

if __name__ == '__main__':
    grid_search()
    alphas = [2]
    epsilons = [0.5]
    tails = [10]
    #test_hiperparameters(alphas,epsilons,tails)
