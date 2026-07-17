from Classificador import OSR_classifier, Discriminator
from Feat_extraction.ResNet18_feature_extraction import ResNet18_cac_feature_extraction
from torch.utils.data import DataLoader, ConcatDataset
from Utils_OpenGan import FeatDataset, Matriz_confusao_osr_dataset_outlier_cumulativa as mc
from Utils import *
from sklearn.metrics import accuracy_score, f1_score
import os, gc
import pandas as pd
from Utils import NOMES

"""Código para implementar o classificador OSR utilizando OpenGan"""
fix_random_seed(42)
device = "cuda:0"

num_classes = 2
bs = 28

model_name = NOMES.RESNET18.value
feats_dir = os.path.join(NOMES.FEATS_DIR.value, "Panicum_plantnet_cac", model_name)

result_dir = os.path.join("/home/alexandreselani/Desktop/OpenGan/Resultados/Panicum_plantnet_cac/", model_name)
os.makedirs(result_dir, exist_ok=True)

def classificacao(dataloader, osr_classifier):
    all_predicts = []
    all_labels = []
    all_scores = []
    for X, y in dataloader:
        X = X.unsqueeze_(-1).unsqueeze_(-1)
        X = X.to(device)
        predicts, outlier_score = osr_classifier.classify(X)
        all_predicts.append(predicts.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_scores.append(outlier_score.detach().cpu())
    
    all_predicts = torch.cat(all_predicts)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    # print(all_scores)
    return all_predicts, all_labels, all_scores

def create_instances(epsilon, num_classes, fold, best_epoch=None):
    #-------------------------parametros do discriminador-----------------
    #numero de canais
    nc = 512

    # Size of feature maps in discriminator
    ndf = 64
    discriminator = Discriminator(nc=nc, ndf=ndf).to(device=device)
    
    path_to_D = os.path.join("/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Panicum_plantnet_cac", model_name, f"Fold+{fold}", 'best_epoch.DNet')

    discriminator.load_state_dict(torch.load(path_to_D))
    discriminator.eval()

    #---------------------Carregando o classificador
    classifier = ResNet18_cac_feature_extraction(num_classes)
    classifier.load_model(torch.load(os.path.join("/home/alexandreselani/Desktop/Experimento_panicum_cac/ResNet18", f"Fold_{fold}", f"ResNet18_Panicum_cac_fold_{fold}_plantnet.pt")))
    classifier.model.eval()

    #classificador + discriminador
    osr = OSR_classifier(classifier=classifier, discriminator=discriminator, epsilon=epsilon)

    return discriminator, classifier, osr

def val():
    epsilons = np.arange(0, 1, 0.01).tolist()
    #epsilons = [0.38]
    epsilons = [round(epsilon, 2) for epsilon in epsilons]
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

    for fold in range(5):
        fold_feat_dir = os.path.join(feats_dir, f"Fold_{fold}")
        fold_results = []

        val_data = torch.load(os.path.join(fold_feat_dir, "val_features.pt"))
        val_dataset = FeatDataset(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    
        discriminator, classifier, osr = create_instances(num_classes=num_classes, fold=fold, epsilon=0)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            osr.set_epsilon(epsilon)

            predicts, labels, outlier_scores = classificacao(val_dataloader, osr)

            metricas = metricasImplementadas(predicts, labels, outlier_scores=outlier_scores, metodo="opengan")
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
        
        matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=val_dir, name=f"epsilon_{epsilon}")

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(val_dir, "Resultados_val.csv"), index=False, float_format="%.3f")
    print(f"Arquivo final salvo em: {result_dir}")

def test():
    epsilons = np.arange(0, 1, 0.01).tolist()
    #epsilons = [0.34]
    epsilons = [round(epsilon, 2) for epsilon in epsilons]
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

    for fold in range(5):
        fold_feat_dir = os.path.join(feats_dir, f"Fold_{fold}")
        fold_results = []

        test_data = torch.load(os.path.join(fold_feat_dir, "test_features.pt"))
        test_dataset = FeatDataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        discriminator, classifier, osr = create_instances(num_classes=num_classes, fold=fold, epsilon=0)

        test_dir = os.path.join(result_dir,"Test")
        os.makedirs(test_dir,exist_ok=True)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            osr.set_epsilon(epsilon)

            predicts, labels, outlier_scores = classificacao(test_dataloader, osr)

            metricas = metricasImplementadas(predicts, labels, outlier_scores=outlier_scores, metodo="opengan")
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
        df_fold.to_csv(os.path.join(test_dir, f"Resultados_Fold_{fold}_test.csv"), index=False, float_format="%.3f")

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

        
        
        matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=test_dir, name=f"epsilon_{epsilon}")

    df = pd.DataFrame(final_data)
    df.to_csv(os.path.join(test_dir, "Resultados_test.csv"), index=False, float_format="%.3f")
    print(f"Arquivo final salvo em: {result_dir}")

if __name__ == "__main__":
    #epsilons = np.arange(0.0, 0.3, 0.01).tolist()
    #model_selection(epsilons)
    #val()
    test()