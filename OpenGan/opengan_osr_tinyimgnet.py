from Classificador import OSR_classifier, Discriminator
from Feat_extraction.ResNet18_64x64_feature_extraction import *
from torch.utils.data import DataLoader, ConcatDataset
from Utils_OpenGan import FeatDataset, Matriz_confusao_osr_dataset_outlier_cumulativa as mc
from Utils import metricasImplementadasV2,metricLogger,fix_random_seed
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os, gc
import pandas as pd
from Utils import NOMES

"""Código para implementar o classificador OSR utilizando OpenGan"""
fix_random_seed(42)
device = "cuda:0"

num_classes = 20
bs = 256

model_name = "ResNet18"

feats_dir = os.path.join(NOMES.FEATS_DIR.value, "Tinyimgnet", model_name)

result_dir = os.path.join("/home/alexandreselani/Desktop/OpenGan/Resultados/Tinyimgnet/", model_name)
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
    ndf = 100
    discriminator = Discriminator(nc=nc, ndf=ndf).to(device=device)
    
    path_to_D = os.path.join("/home/alexandreselani/Desktop/OpenGan/OpenGAN-IC/Experimentos/Tinyimgnet", model_name, f"Fold+{fold}", 'best_epoch.DNet')

    discriminator.load_state_dict(torch.load(path_to_D))
    discriminator.eval()

    #---------------------Carregando o classificador
    classifier = ResNet18_64x64_feature_extraction(num_classes)
    classifier.load_model(torch.load(os.path.join(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/Split_{fold}/ResNet18_TinyImageNet_split_{fold}.pt")))
    classifier.model.eval()

    #classificador + discriminador
    osr = OSR_classifier(classifier=classifier, discriminator=discriminator, epsilon=epsilon)

    return discriminator, classifier, osr

def val():
    epsilons = np.arange(0, 1, 0.01).tolist()
    #epsilons = [0.38]
    epsilons = [round(epsilon, 2) for epsilon in epsilons]

    metric_logger = metricLogger(epsilons,5,os.path.join(result_dir,"Val"),mc_column_names=["Unknown"] + [str(i) for i in range(num_classes)])

    for fold in range(5):
        fold_feat_dir = os.path.join(feats_dir, f"Split_{fold}")
        fold_results = []

        val_data = torch.load(os.path.join(fold_feat_dir, "val_features.pt"))
        val_dataset = FeatDataset(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    
        discriminator, classifier, osr = create_instances(num_classes=num_classes, fold=fold, epsilon=0)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            osr.set_epsilon(epsilon)

            predicts, labels, outlier_scores = classificacao(val_dataloader, osr)

            metricas = metricasImplementadasV2(predicts, labels, outlier_scores=outlier_scores, metodo="opengan")
            metricas = metricas._metricas()

            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)

    metric_logger.aggregate("Val.csv")
           
def test():
    epsilons = np.arange(0, 1, 0.01).tolist()
    #epsilons = [0.38]
    epsilons = [round(epsilon, 2) for epsilon in epsilons]

    metric_logger = metricLogger(epsilons,5,os.path.join(result_dir,"Test"),mc_column_names=["Unknown"] + [str(i) for i in range(num_classes)])

    for fold in range(5):
        fold_feat_dir = os.path.join(feats_dir, f"Split_{fold}")
        fold_results = []

        test_data = torch.load(os.path.join(fold_feat_dir, "test_features.pt"))
        test_dataset = FeatDataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
        discriminator, classifier, osr = create_instances(num_classes=num_classes, fold=fold, epsilon=0)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            osr.set_epsilon(epsilon)

            predicts, labels, outlier_scores = classificacao(test_dataloader, osr)

            metricas = metricasImplementadasV2(predicts, labels, outlier_scores=outlier_scores, metodo="opengan")
            metricas = metricas._metricas()

            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)

    metric_logger.aggregate("Test.csv")

if __name__ == "__main__":
    #epsilons = np.arange(0.0, 0.3, 0.01).tolist()
    #model_selection(epsilons)
    val()
    test()