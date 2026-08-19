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
import torch
from pytorch_ood.detector import OpenMax
import numpy as np
import os
import gc
from Utils import fix_random_seed,NOMES,metricasImplementadasV2,metricLogger
from Datasets import TinyImageNet_loader
from Modelos import ResNet18_tinyimgnet
seed = 42
fix_random_seed(seed)

device = "cuda:0"

DATA_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"
RESNET_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18"
N_SPLITS = 5
NUM_KNOWN_CLASSES = 20


def load_split_model(split):
    model = ResNet18_tinyimgnet(num_classes=NUM_KNOWN_CLASSES, weights=None)
    weights_path = os.path.join(RESNET_DIR, f"Split_{split}", f"ResNet18_TinyImageNet_split_{split}.pt")
    model.load_state_dict(torch.load(weights_path))
    model.to(device=device)
    model.eval()
    return model


def fit_loader(data_manager, split):
    # Troca o transform do loader de treino publico para o eval_transform (sem augmentation)
    # antes de ajustar as distribuicoes de Weibull do OpenMax -- mesma escolha feita para o
    # Panicum, que usava o transform de validacao ao chamar detector.fit().
    loader = data_manager.get_train_loader(split)
    _, eval_transform = data_manager._get_transforms(split)
    loader.dataset.transform = eval_transform
    return loader


def collect_targets(loader):
    all_targets = np.array([])
    for (_, y) in loader:
        all_targets = np.append(all_targets, y.detach().cpu().numpy())
    return all_targets


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
max_tailsize = 200
step_tail = 50
tails = list(range(min_tailsize, max_tailsize+1, step_tail))
alphas = [1,5,10,15,20]
epsilons = np.arange(0,1,0.2)

def grid_search():
    nomeDataset = "TinyImageNet"
    output_dir = f"/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/{nomeDataset}/Val/"
    os.makedirs(output_dir, exist_ok=True)

    mc_column_names = ["Unknown"] + [str(i) for i in range(NUM_KNOWN_CLASSES)]

    data_manager = TinyImageNet_loader(data_dir=DATA_DIR, splits_dir=SPLITS_DIR, batch_size=32, image_size=64)

    

    for alpha in alphas:
        for epsilon in epsilons:

            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            organized_dir = os.path.join(output_dir, pasta)
            os.makedirs(organized_dir, exist_ok=True)

            registra_metricas = metricLogger(tails,N_SPLITS,organized_dir,mc_column_names=mc_column_names,mc_title=nomeDataset,predict_unknown_value=0)

            for split in range(N_SPLITS):
                gc.collect()
                torch.cuda.empty_cache()

                model = load_split_model(split)

                train_dataloader = fit_loader(data_manager, split)
                val_dataloader = data_manager.get_val_loader(split)

                all_targets = collect_targets(val_dataloader)

                for tail in tails:
                    print(f"Split:{split} tail {tail} alpha {alpha}, epsilon {epsilon}")
                    gc.collect()
                    detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                    detector.fit(train_dataloader, device=device)

                    metricas, predicts, targets_val = test(val_dataloader, detector)
                    registra_metricas.update(metricas,split,tail)
                    registra_metricas.update_mc(tail,predicts,targets_val,all_targets)

                    del detector

                del model

            registra_metricas.aggregate("gridsearch.csv")

def run_test_evaluation():
    nomeDataset = "TinyImageNet"
    output_dir = f"/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/{nomeDataset}/Test/"
    os.makedirs(output_dir, exist_ok=True)

    mc_column_names = ["Unknown"] + [str(i) for i in range(NUM_KNOWN_CLASSES)]

    data_manager = TinyImageNet_loader(data_dir=DATA_DIR, splits_dir=SPLITS_DIR, batch_size=32, image_size=64)


    for alpha in alphas:
        for epsilon in epsilons:

            pasta = f"alpha_{alpha}/epsilon_{epsilon}/"
            organized_dir = os.path.join(output_dir, pasta)
            os.makedirs(organized_dir, exist_ok=True)

            registra_metricas = metricLogger(tails,N_SPLITS,organized_dir,mc_column_names=mc_column_names,mc_title=nomeDataset,predict_unknown_value=0)

            for split in range(N_SPLITS):
                gc.collect()
                torch.cuda.empty_cache()

                model = load_split_model(split)

                train_dataloader = fit_loader(data_manager, split)
                test_dataloader = data_manager.get_test_loader(split)

                all_targets = collect_targets(test_dataloader)

                for tail in tails:
                    print(f"Split:{split} tail {tail} alpha {alpha}, epsilon {epsilon}")
                    gc.collect()
                    detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1, epsilon=epsilon)
                    detector.fit(train_dataloader, device=device)

                    metricas, predicts, targets_test = test(test_dataloader, detector)

                    registra_metricas.update(metricas,split,tail)
                    registra_metricas.update_mc(tail,predicts,targets_test,all_targets)

                    del detector

                del model

            registra_metricas.aggregate("Test.csv")


if __name__ == '__main__':
    grid_search()
    run_test_evaluation()
