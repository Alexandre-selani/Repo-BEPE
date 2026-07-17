
from torch.utils.data import DataLoader, ConcatDataset
from utils import gather_outputs, find_anchor_means, Matriz_confusao_osr_dataset_outlier_cumulativa as mc
from Utils import *
from Modelos import ResNet18_cac
from sklearn.metrics import accuracy_score, f1_score
import os, gc
import pandas as pd
from Utils import NOMES
from Datasets import Panicum_halfsize_loader
fix_random_seed(42)
device = "cuda:0"

num_classes = 2
bs = 28

model_name = NOMES.RESNET18.value
result_dir = os.path.join("/home/alexandreselani/Desktop/cac-openset-Dimity/Results/Panicum_plantnet_cac/", model_name)
os.makedirs(result_dir, exist_ok=True)


def val(epsilons):
    
    #epsilons = [0.38]
    epsilons = [round(epsilon, 3) for epsilon in epsilons]
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
    val_dir = os.path.join(result_dir,"Val")
    os.makedirs(val_dir,exist_ok=True)

    data_manager = Panicum_halfsize_loader(bs=32)

    for fold in range(5):
        fold_results = []

        trainloader = data_manager.load_train(fold,NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value)
        valloader = data_manager.load_val(fold,NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value)
        
        model = ResNet18_cac(num_classes).to(device)
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_panicum_cac/ResNet18/Fold_{fold}/ResNet18_Panicum_cac_fold_{fold}_plantnet.pt"))
        
        anchor_means = find_anchor_means(model,trainloader,device,num_classes)
        model.set_anchors(anchor_means)

        logits, distances, labels = gather_outputs(model,valloader,device)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            
            predicts, min_scores, outlier_scores = model.predict_by_distance(epsilon, distances)

            metricas = metricasImplementadas(predicts, labels, outlier_scores=-min_scores, metodo="opengan")
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
                matriz = mc(predicts, labels, labels, [], ["Panicum", "Ground", "Healthy"])
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
        
        cm_dir = os.path.join(val_dir,"confusion_matrices")
        os.makedirs(cm_dir,exist_ok=True)
        matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=cm_dir, name=f"epsilon_{epsilon}")

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(val_dir, "Resultados_val.csv"), index=False, float_format="%.3f")
    print(f"Arquivo final salvo em: {result_dir}")

def test(epsilons):
    
    #epsilons = [0.38]
    epsilons = [round(epsilon, 3) for epsilon in epsilons]
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
    val_dir = os.path.join(result_dir,"Test")
    os.makedirs(val_dir,exist_ok=True)

    data_manager = Panicum_halfsize_loader(bs=32)

    for fold in range(5):
        fold_results = []

        trainloader = data_manager.load_train(fold,NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value)
        testloader = data_manager.load_test(fold,NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value)
        
        model = ResNet18_cac(num_classes).to(device)
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_panicum_cac/ResNet18/Fold_{fold}/ResNet18_Panicum_cac_fold_{fold}_plantnet.pt"))
        
        anchor_means = find_anchor_means(model,trainloader,device,num_classes)
        model.set_anchors(anchor_means)

        logits, distances, labels = gather_outputs(model,testloader,device)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            
            predicts, min_scores, outlier_scores = model.predict_by_distance(epsilon, distances)

            metricas = metricasImplementadas(predicts, labels, outlier_scores=-min_scores, metodo="opengan")
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
                matriz = mc(predicts, labels, labels, [], ["Panicum", "Ground", "Healthy"])
                matriz.computa_matriz()
                matrizes_confusao_acumulada[epsilon] = matriz
            else:
                matrizes_confusao_acumulada[epsilon].set_data(predicts, labels, labels)
                matrizes_confusao_acumulada[epsilon].computa_matriz()

        df_fold = pd.DataFrame(fold_results)
        df_fold.to_csv(os.path.join(val_dir, f"Resultados_Fold_{fold}_Test.csv"), index=False, float_format="%.3f")

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
        
        cm_dir = os.path.join(val_dir,"confusion_matrices")
        os.makedirs(cm_dir,exist_ok=True)
        matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=cm_dir, name=f"epsilon_{epsilon}")

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(val_dir, "Resultados_test.csv"), index=False, float_format="%.3f")
    print(f"Arquivo final salvo em: {result_dir}")

if __name__ == "__main__":
    epsilons_val = np.arange(0.0, 0.4, 0.001).tolist()
    epsilons_test = np.arange(0.0, 0.1, 0.001).tolist()
    #model_selection(epsilons)
    #val(epsilons_val)
    test(epsilons_test)