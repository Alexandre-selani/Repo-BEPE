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

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.dataset.img import PanicumDataset_pytorchOOD
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import PlainCNN_panicum
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, metricasImplementadas,AnaliseGrafica_OpenMax,Matriz_confusao_osr_dataset_outlier as mc
from pytorch_ood.utils.aux_dataset import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
seed = 777
fix_random_seed(seed)

device = "cuda:0"

ood_metrics = OODMetrics()


def test(test_loader,detector):
    predicts=[]
    labels=[]

    for X, y in test_loader:
        
        #score eh a ativacao de todas as classes apos a openmax
        with torch.no_grad():
            score = detector(X.to(device))
            print(score)
            max_values, predicted = torch.max(score, dim=1)
            predict = torch.where(max_values >= detector.epsilon, predicted, torch.zeros_like(predicted))

        
        predicts.append(predict.detach().cpu())
        labels.append(y.detach().cpu())
        
    
    predicts = torch.cat(predicts,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    #ood_metrics.update(score[:,0],y)
    metricas = metricasImplementadas(predict=predicts, label=labels)

    #print(ood_metrics.compute())
    return metricas._metricas()
    
def train(train_loader,model,criterion,optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/(batch_idx+1), correct/total

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

def validation(val_loader,model,criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return val_loss/(batch_idx+1), correct/total

def confusion_matrix(test_loader,targets_original,nome_classes_originais,UUC_classes,detector,dir):
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
    matriz_confusao.exibe_matriz(dir=dir)

def main():
    nomeDataset = "panicum"

    analiseGrafica = AnaliseGrafica_OpenMax(nomeDataset)

    weights = ResNet50_Weights.IMAGENET1K_V2
    transforms = weights.transforms()

    imagens_kkc = ImageFolder(root='/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Todas/KKC',transform=transforms)

    imagens_uuc = ImageFolder(root='/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Todas/UUC',transform=transforms,target_transform=ToUnknown())
    
    

    bs = 64
    lr = 0.0002
    epochs=35
    alpha=2
    min_tail=0
    max_tail=1000

    
    for epsilon in [0.3,0.35,0.4]:
        f1s = {}
        f1s_std = {}

        accs = {}
        accs_std = {}

        uuc_accs = {}
        uuc_accs_std = {}

        inners = {}
        inners_std = {}

        outers = {}
        outers_std = {}

        halfpoints = {}
        halfpoints_std = {}

        val_accs = {}
        val_accs_std = {}

        val_losses = {}
        for tail in range(min_tail,max_tail+1,100):
            
            print(f"TAIL = {tail}")
            all_f1 = []
            all_acc=[]
            all_uuc_acc=[]
            all_inner=[]
            all_outer=[]
            all_halfpoint=[]
            all_val_acc=[]
            all_val_loss=[]

            kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

            for i, (train_index, test_index) in enumerate(kfold.split(imagens_kkc.imgs, imagens_kkc.targets)):
                
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
                model.fc = nn.Linear(model.fc.in_features, 2).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

                print(f"Fold {i}:")
                train_imgs = Subset(imagens_kkc,train_index)
                train_imgs,val_imgs = validation_split(0.20,train_imgs)

                test_imgs = Subset(imagens_kkc,test_index)
                
                
                test_imgs = test_imgs+imagens_uuc
                print("tamanho treino ",len(train_imgs),"\ntamanho validacao ",len(val_imgs),"\ntamanho teste:",len(test_imgs))

                train_loader = DataLoader(train_imgs,batch_size=bs,shuffle=True,num_workers=4,pin_memory=True)
                test_loader = DataLoader(test_imgs,batch_size=bs,shuffle=False,num_workers=4,pin_memory=True)
                val_loader = DataLoader(val_imgs,batch_size=bs,shuffle=False,num_workers=4,pin_memory=True)

                all_targets = np.array([])

                for (x, y) in test_loader:
                    for target in y:
                        all_targets= np.append(all_targets,target.detach().cpu())
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

                detector = OpenMax(model, tailsize=tail, alpha=alpha, euclid_weight=1,epsilon=epsilon)

                t_loss,t_acc=[],[]
                v_loss,v_acc=[],[]

                for epoch in range(epochs):
                    
                    train_loss, train_acc = train(train_loader, model, criterion, optimizer)
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
                    val_loss, val_acc = validation(val_loader,model,criterion)

                    t_loss.append(train_loss)
                    t_acc.append(train_acc)
                    v_loss.append(val_loss)
                    v_acc.append(val_acc)


                    #scheduler.step()
                    #print(metricas)
                    #analiseGrafica.addEpoch(metricas,epoch,train_loss=train_loss,train_acc=train_acc,val_loss=val_loss,val_acc=val_acc)
                
                plt.figure(figsize=(8,5))
                e = range(0,epochs)
                plt.plot(e, t_loss, 'red', label='Train Loss')
                plt.plot(e, v_loss, 'orange', label='Val Loss')
                plt.plot(e, t_acc, 'blue', label='Train Acc')
                plt.plot(e, v_acc, 'purple', label='Val Acc')

                plt.xlabel('Épocas')
                plt.xticks(e)
                plt.ylabel('Valor')
                plt.title('Evolução de Loss e Acurácia')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                import os

                dir = f"/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum/Resultados OpenMax/Epsilon {epsilon}/tailsize {tail}/"
                os.makedirs(dir, exist_ok=True)

                plt.savefig(dir+f"Fold {i}.png")

                detector.fit(train_loader, device=device)
                metricas = test(test_loader,detector)
                
                if(i==0):
                    confusion_matrix(test_loader,all_targets,["Panicum","Solo","Milho"],[],detector,dir=dir)
                print(f"VALIDATION || Epoch {epoch+1}/{epochs} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}| lr = {optimizer.param_groups[0]['lr']}")
                print(metricas)
                all_f1.append(metricas["F1 macro"])
                all_acc.append(metricas["accuracy"][0])
                all_uuc_acc.append(metricas["UUC Accuracy"][0])
                all_inner.append(metricas["inner metric"][0])
                all_outer.append(metricas["outer metric"][0])
                all_halfpoint.append(metricas["halfpoint"][0])

                torch.cuda.empty_cache()


                #analiseGrafica.mostraGrafico(tail=tail,alpha=alpha,epsilon=epsilon,fold=i)

            f1s[tail]        = np.array(all_f1).mean()
            f1s_std[tail]    = np.array(all_f1).std()

            accs[tail]       = np.array(all_acc).mean()
            accs_std[tail]   = np.array(all_acc).std()

            inners[tail]     = np.array(all_inner).mean()
            inners_std[tail] = np.array(all_inner).std()

            outers[tail]     = np.array(all_outer).mean()
            outers_std[tail] = np.array(all_outer).std()

            uuc_accs[tail]       = np.array(all_uuc_acc).mean()
            uuc_accs_std[tail]   = np.array(all_uuc_acc).std()

            halfpoints[tail]        = np.array(all_halfpoint).mean()
            halfpoints_std[tail]    = np.array(all_halfpoint).std()
            
        
        df = pd.DataFrame({
        "tail": list(f1s.keys()),
        
        "f1_macro_medio": list(f1s.values()),
        #"f1_macro_std": list(f1s_std.values()),
        
        "acc_medio": list(accs.values()),
        #"acc_std": list(accs_std.values()),
        
        "uuc_acc_medio": list(uuc_accs.values()),
        #"uuc_acc_std": list(uuc_accs_std.values()),
        
        "inner_medio": list(inners.values()),
        #"inner_std": list(inners_std.values()),
        
        "outer_medio": list(outers.values()),
        #"outer_std": list(outers_std.values()),
        
        "halfpoint_medio": list(halfpoints.values()),
        #"halfpoint_std": list(halfpoints_std.values())
    })

        df.to_csv(f"/home/alexandreselani/Desktop/pytorch-ood/pytorch-ood/experimento_panicum/Resultados OpenMax/Resultados_grid_search_panicum_tail_0_1000_epsilon_{epsilon}.csv", index=False)

        

    
if __name__ == '__main__':
    main()

