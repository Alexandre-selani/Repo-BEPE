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
from Utils import metricasImplementadas,NOMES,Matriz_confusao_osr_dataset_outlier_cumulativa as mc
import pandas as pd

import pandas
import os
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
MODEL = "AlexNet"
DATASET = "dataset-1"
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
    results_by_epsilon = {epsilon:  {"epsilon": [],
                "f1":[],
                "acc": [],
                "uuc_acc": [],
                "inner": [],
                "outer": [],
                "half": [],
                "auroc": []} 
        for epsilon in epsilons}
    
    matrizes_confusao_acumulada = {epsilon: None for epsilon in epsilons}

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

            metricas = metricasImplementadas(predicts,labels,outlier_scores=scores,metodo="opengan")
            metricas = metricas._metricas()

            results_by_epsilon[epsilon]['f1'].append(metricas["F1 macro"])
            results_by_epsilon[epsilon]['acc'].append(metricas["accuracy"][0])
            results_by_epsilon[epsilon]['uuc_acc'].append(metricas["UUC Accuracy"][0])
            results_by_epsilon[epsilon]['inner'].append(metricas["inner metric"][0])
            results_by_epsilon[epsilon]['outer'].append(metricas["outer metric"][0])
            results_by_epsilon[epsilon]['half'].append(metricas["halfpoint"][0])
            results_by_epsilon[epsilon]['auroc'].append(metricas['auroc'])
            
            fold_results.append({
                "epsilon": epsilon,
                "f1_macro": metricas["F1 macro"],
                "acc": metricas["accuracy"][0],
                "uuc_acc": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas['auroc']
            })

            if matrizes_confusao_acumulada[epsilon] is None:
                matriz = mc(predicts, labels, labels, [], ["Ceratocystis", "Ground", "Healthy"])
                matriz.computa_matriz()
                matrizes_confusao_acumulada[epsilon] = matriz
            else:
                matrizes_confusao_acumulada[epsilon].set_data(predicts, labels, labels)
                matrizes_confusao_acumulada[epsilon].computa_matriz()

        df_fold = pd.DataFrame(fold_results)
        df_fold.to_csv(os.path.join(val_dir, f"Resultados_Fold_{fold}_Val.csv"), index=False, float_format="%.3f")

    final_data = []
    
    for epsilon in sorted(results_by_epsilon.keys()):
        metrics = results_by_epsilon[epsilon]
        row = {
            "epsilon": epsilon,
            "f1_macro_medio": np.mean(metrics['f1']),
            "f1_macro_std": np.std(metrics['f1']),
            "acc_medio": np.mean(metrics['acc']),
            "acc_std": np.std(metrics['acc']),
            "uuc_acc_medio": np.mean(metrics['uuc_acc']),
            "uuc_acc_std": np.std(metrics['uuc_acc']),
            "inner_medio": np.mean(metrics['inner']),
            "inner std": np.std(metrics["inner"]),
            "outer_medio": np.mean(metrics['outer']),
            "outer_std": np.std(metrics["outer"]),
            "halfpoint_medio": np.mean(metrics['half']),
            "halfpoint_std": np.std(metrics['half']),
            "auroc_media": np.mean(metrics['auroc']),
            "auroc_std": np.std(metrics['auroc'])
        }
        final_data.append(row)
        
        matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=os.path.join(val_dir,"matrizes"), name=f"epsilon_{epsilon}")

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(val_dir, "Resultados_val.csv"), index=False, float_format="%.3f")
    print(f"Arquivo final salvo em: {result_dir}")
    

def test(epsilons):
    test_dir = os.path.join(result_dir,"Test")
    os.makedirs(test_dir,exist_ok=True)
    results_by_epsilon = {epsilon:  {"epsilon": [],
                "f1":[],
                "acc": [],
                "uuc_acc": [],
                "inner": [],
                "outer": [],
                "half": [],
                "auroc": []} 
        for epsilon in epsilons}
    
    matrizes_confusao_acumulada = {epsilon: None for epsilon in epsilons}

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

            metricas = metricasImplementadas(predicts,labels,outlier_scores=scores,metodo="opengan")
            metricas = metricas._metricas()

            results_by_epsilon[epsilon]['f1'].append(metricas["F1 macro"])
            results_by_epsilon[epsilon]['acc'].append(metricas["accuracy"][0])
            results_by_epsilon[epsilon]['uuc_acc'].append(metricas["UUC Accuracy"][0])
            results_by_epsilon[epsilon]['inner'].append(metricas["inner metric"][0])
            results_by_epsilon[epsilon]['outer'].append(metricas["outer metric"][0])
            results_by_epsilon[epsilon]['half'].append(metricas["halfpoint"][0])
            results_by_epsilon[epsilon]['auroc'].append(metricas['auroc'])
            
            fold_results.append({
                "epsilon": epsilon,
                "f1_macro": metricas["F1 macro"],
                "acc": metricas["accuracy"][0],
                "uuc_acc": metricas["UUC Accuracy"][0],
                "inner_metric": metricas["inner metric"][0],
                "outer_metric": metricas["outer metric"][0],
                "halfpoint": metricas["halfpoint"][0],
                "auroc": metricas['auroc']
            })

            if matrizes_confusao_acumulada[epsilon] is None:
                matriz = mc(predicts, labels, labels, [], ["Ceratocystis", "Ground", "Healthy"])
                matriz.computa_matriz()
                matrizes_confusao_acumulada[epsilon] = matriz
            else:
                matrizes_confusao_acumulada[epsilon].set_data(predicts, labels, labels)
                matrizes_confusao_acumulada[epsilon].computa_matriz()

        df_fold = pd.DataFrame(fold_results)
        df_fold.to_csv(os.path.join(test_dir, f"Resultados_Fold_{fold}_Test.csv"), index=False, float_format="%.3f")

    final_data = []
    
    for epsilon in sorted(results_by_epsilon.keys()):
        metrics = results_by_epsilon[epsilon]
        row = {
            "epsilon": epsilon,
            "f1_macro_medio": np.mean(metrics['f1']),
            "f1_macro_std": np.std(metrics['f1']),
            "acc_medio": np.mean(metrics['acc']),
            "acc_std": np.std(metrics['acc']),
            "uuc_acc_medio": np.mean(metrics['uuc_acc']),
            "uuc_acc_std": np.std(metrics['uuc_acc']),
            "inner_medio": np.mean(metrics['inner']),
            "inner std": np.std(metrics["inner"]),
            "outer_medio": np.mean(metrics['outer']),
            "outer_std": np.std(metrics["outer"]),
            "halfpoint_medio": np.mean(metrics['half']),
            "halfpoint_std": np.std(metrics['half']),
            "auroc_media": np.mean(metrics['auroc']),
            "auroc_std": np.std(metrics['auroc'])
        }
        final_data.append(row)
        
        matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=os.path.join(test_dir,"matrizes"), name=f"epsilon_{epsilon}")

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(test_dir, "Resultados_test.csv"), index=False, float_format="%.3f")
    print(f"Arquivo final salvo em: {result_dir}")
    

thresholds = np.arange(0,1,0.01)
train()
val(thresholds)
test(thresholds)


