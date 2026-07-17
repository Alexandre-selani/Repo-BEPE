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
from Modelos.LeNet_backbone import LeNet
from Utils import fix_random_seed,metricasImplementadas,NOMES
from Datasets import Mnist_omni_loader

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
    output_dir = "/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_mnist_omniglot/Resultados OpenMax/LeNet"
    os.makedirs(output_dir, exist_ok=True)
    
    model = LeNet(num_classes=10)
    model.load_state_dict(torch.load(NOMES.LENET_MNIST_OMNI.value))
    model.to(device=device)
    bs = 256


    data = Mnist_omni_loader(bs,transform=NOMES.LENET_MNIST_OMNI_TRANSFORMS.value)

    train_dataloader = data.load_train()
    grid_search_val_dataloader = data.load_gridsearch()
    test_dataloader = data.load_test()

    # grid search
    min_tailsize = 300
    max_tailsize = 1000
    step_tail = 100
    min_alpha = 1
    max_alpha = 3
    
    epsilons = np.arange(0.5, 1, 0.2).tolist()

    melhores_hiperparametros = {'alpha':None,
                                'epsilon':None,
                                'tail':None}
    melhor_f1=-1

    for alpha in range(min_alpha,max_alpha+1):

        for epsilon in epsilons:
            
            epsilon = round(epsilon,1)
            matrizes_confusao = [None for _ in range(min_tailsize, max_tailsize+1, step_tail)]
            #print(matrizes_confusao)
        
            torch.cuda.empty_cache()
            gc.collect()
            
            model.eval()

            results_by_tail = {}
            
            all_targets = np.array([])

            for (_, y) in grid_search_val_dataloader:
                for target in y:
                    all_targets= np.append(all_targets,target.detach().cpu())

            for idx,tail in enumerate(range(min_tailsize, max_tailsize+1, step_tail)):
                
                print(alpha, tail, epsilon)
                # Fit e Teste
                # Nota: O Fit ainda pode ser demorado, mas economizamos o load do modelo
                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                
                # Ajuste (Fit) - Geralmente precisa passar os dados de treino para calcular os centros/weibulls
                detector.fit(train_dataloader, device=device)
                
                metricas,predicts,targets_test = test(grid_search_val_dataloader, detector)
                
                matriz = mc(predicts,targets_test,all_targets,[],["Omniglot",0,1,2,3,4,5,6,7,8,9])
                matriz.computa_matriz()
                matrizes_confusao[idx] = matriz

                if metricas["F1 macro"] > melhor_f1:
                    melhores_hiperparametros["alpha"] = alpha
                    melhores_hiperparametros["epsilon"] = epsilon
                    melhores_hiperparametros["tail"] = tail
                    melhor_f1 = metricas["F1 macro"]

                # Guardar métricas deste fold para este tail
                results_by_tail[tail] = {
                "tail": tail,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]
                }


                del detector
                gc.collect()
                torch.cuda.empty_cache()
            
            final_data = []

            for tail in sorted(results_by_tail.keys()):
                metrics = results_by_tail[tail]
                final_data.append(metrics)

            df = pd.DataFrame(final_data)

            
            # Nome dinâmico correto
            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            filename_csv = f"Resultados_grid_mnist_omniglot_tail_{min_tailsize}_{max_tailsize}_eps_{epsilon}_alpha_{alpha}.csv"
            organized_dir = os.path.join(output_dir, pasta)

            os.makedirs(organized_dir, exist_ok=True)

            csv_path = os.path.join(organized_dir,filename_csv)
            
            for idx,m in enumerate(matrizes_confusao):
                m.exibe_matriz(dir=organized_dir,name=f"tail_{idx*step_tail}")

            df.to_csv(csv_path, index=False,float_format="%.3f")
            print(f"Arquivo salvo: {csv_path}")

    
    #RESULTADOS DO TESTE COM O MODELO SELECIONADO
    melhor_tail = melhores_hiperparametros["tail"]
    melhor_epsilon = melhores_hiperparametros["epsilon"]
    melhor_alpha = melhores_hiperparametros["alpha"]

    detector = OpenMax(model, tailsize=melhor_tail,
                              alpha=melhor_alpha, euclid_weight=1, 
                              epsilon=melhor_epsilon)       
    
    # Ajuste (Fit) - Geralmente precisa passar os dados de treino para calcular os centros/weibulls
    detector.fit(train_dataloader, device=device)
    
    metricas,predicts,targets_test = test(test_dataloader, detector)

    results_by_tail = {}
            
    all_targets = np.array([])

    for (_, y) in test_dataloader:
        for target in y:
            all_targets= np.append(all_targets,target.detach().cpu())
    
    results_by_tail[melhor_tail] = {
                "tail": melhor_tail,
                "f1_macro": metricas["F1 macro"],
                "accuracy": metricas["accuracy"][0],
                "uuc_accuracy": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas["auroc"]
                }
    
    final_data = []

    for tail in sorted(results_by_tail.keys()):
        metrics = results_by_tail[tail]
        final_data.append(metrics)

    df = pd.DataFrame(final_data)

    matriz = mc(predicts,targets_test,all_targets,[],["Unknown",0,1,2,3,4,5,6,7,8,9])
    matriz.computa_matriz()

    
    filename_csv = f"melhor_modelo_teste_{min_tailsize}_{max_tailsize}_eps_{melhores_hiperparametros['epsilon']}_alpha_{melhores_hiperparametros['alpha']}.csv"
    organized_dir = os.path.join(output_dir, filename_csv)
    
    matriz.exibe_matriz(dir=output_dir,name=f"melhor_modelo.png")

    df.to_csv(organized_dir, index=False,float_format="%.3f")
    print(f"Arquivo salvo: melhor resultado")
    print(f"Melhores hiperparametros \n {melhores_hiperparametros}")
    
if __name__ == '__main__':
    main()



