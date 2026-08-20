
from torch.utils.data import DataLoader, ConcatDataset
from utils import gather_outputs, find_anchor_means, Matriz_confusao_osr_dataset_outlier_cumulativa as mc
from Utils import *
from Modelos import ResNet18_tinyimgnet_cac
from sklearn.metrics import accuracy_score, f1_score
import os, gc
import pandas as pd
from Utils import NOMES,metricLogger
from Datasets import TinyImageNet_loader
fix_random_seed(42)
device = "cuda:0"

num_classes = 20
bs = 256

model_name = NOMES.RESNET18.value
result_dir = os.path.join("/home/alexandreselani/Desktop/cac-openset-Dimity/Results/Tinyimgnet_cac/", model_name)
os.makedirs(result_dir, exist_ok=True)

column_names = ["Unknown"] + [str(i) for i in range(num_classes)]

def val(epsilons):
    
    #epsilons = [0.38]
    epsilons = [round(epsilon, 3) for epsilon in epsilons]
    
    val_dir = os.path.join(result_dir,"Val")
    os.makedirs(val_dir,exist_ok=True)

    data_manager = TinyImageNet_loader()
    metric_logger = metricLogger(epsilons,5,val_dir,mc_column_names=column_names)
    for fold in range(5):
        

        trainloader = data_manager.get_train_loader(fold,data_manager.eval_transforms[fold])
        valloader = data_manager.get_val_loader(fold,data_manager.eval_transforms[fold])
        
        model = ResNet18_tinyimgnet_cac(num_classes).to(device)
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/Split_{fold}/ResNet18_TinyImageNet_cac_split_{fold}.pt"))
        
        anchor_means = find_anchor_means(model,trainloader,device,num_classes)
        model.set_anchors(anchor_means)

        logits, distances, labels = gather_outputs(model,valloader,device)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            
            predicts, min_scores, outlier_scores = model.predict_by_distance(epsilon, distances)

            metricas = metricasImplementadasV2(predicts, labels, outlier_scores=-min_scores, metodo="opengan")
            metricas = metricas._metricas()

            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)

    metric_logger.aggregate("Val.csv")


def test(epsilons):
    
    epsilons = [round(epsilon, 3) for epsilon in epsilons]
        
    test_dir = os.path.join(result_dir,"Test")
    os.makedirs(test_dir,exist_ok=True)

    data_manager = TinyImageNet_loader()
    metric_logger = metricLogger(epsilons,5,test_dir,mc_column_names=column_names)
    for fold in range(5):
        

        trainloader = data_manager.get_train_loader(fold,data_manager.eval_transforms[fold])
        test_loader = data_manager.get_test_loader(fold,data_manager.eval_transforms[fold])
        
        model = ResNet18_tinyimgnet_cac(num_classes).to(device)
        model.load_state_dict(torch.load(f"/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/Split_{fold}/ResNet18_TinyImageNet_cac_split_{fold}.pt"))
        
        anchor_means = find_anchor_means(model,trainloader,device,num_classes)
        model.set_anchors(anchor_means)

        logits, distances, labels = gather_outputs(model,test_loader,device)

        for epsilon in epsilons:
            print(f"fold {fold}, epsilon {epsilon}")
            
            predicts, min_scores, outlier_scores = model.predict_by_distance(epsilon, distances)

            metricas = metricasImplementadasV2(predicts, labels, outlier_scores=-min_scores, metodo="opengan")
            metricas = metricas._metricas()

            metric_logger.update(metricas,fold,epsilon)
            metric_logger.update_mc(epsilon,predicts,labels,labels)

    metric_logger.aggregate("Test.csv")

if __name__ == "__main__":
    epsilons_val = np.arange(0.0, 1, 0.001).tolist()
    epsilons_test = np.arange(0.0, 1, 0.001).tolist()
    #model_selection(epsilons)
    #val(epsilons_val)
    test(epsilons_test)