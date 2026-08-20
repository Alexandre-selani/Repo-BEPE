"""
    Helper functions for training and evaluation.

    progress_bar and format_time function was taken from https://github.com/kuangliu/pytorch-cifar which mimics xlua.progress

    Dimity Miller, 2020
"""

import os
import sys
import time
import math
import numpy as np
import torch

from networks import openSetClassifier

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 84

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def find_anchor_means(net, dataloader,device,num_classes):
    
    all_logits = []
    all_targets = []
    all_predicts = []
    for X,y in dataloader:
        X = X.to(device)

        net.skip_distances = True
        net.eval()

        with torch.no_grad():
            logits = net(X)
            _,predicts = torch.max(logits,1)
           
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        all_predicts.append(predicts.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_predicts = torch.cat(all_predicts)

    means = torch.zeros(num_classes,num_classes,dtype=torch.float64)

    for cl in range(num_classes):
        mask = (all_targets == cl) & (all_predicts == cl)
        x = all_logits[mask]
        x = np.squeeze(x)
        
        means[cl] = torch.mean(x, dim = 0)

    #print(means)
    return means

def gather_outputs(net,dataloader,device):
    """Retorna logits, distancias e targets"""
    all_logits = []
    all_targets = []
    all_distances = []

    for X,y in dataloader:
        X = X.to(device)
        y = y.to(device)

        net.eval()
        net.skip_distances = False

        with torch.no_grad():
            logits,distances = net(X)
            
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        all_distances.append(distances.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_distances = torch.cat(all_distances)
    
    return all_logits,all_distances,all_targets


def SoftmaxTemp(logits, T = 1):
    num = torch.exp(logits/T) 
    denom = torch.sum(torch.exp(logits/T), 1).unsqueeze(1) 
    return num/denom 


import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.colors as mcolors
import os.path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.colors as mcolors
import os.path

class Matriz_confusao_osr_dataset_outlier_cumulativa:
    def __init__(self,predict,target_test,target_original,UUC_classes,col_labels):
        self.predict = predict+1
        self.target_test = target_test+1#eh preciso somar um pois podem haver targets = -1 no caso de usar um dataset inteiro como deconhecido junto com certas classes desconhecidas (como mnist + omniglot com omniglot e classes 7,8,9 como desconhecidas)
        self.target_original=target_original+1
        self.UUC_classes = np.array(UUC_classes)+1
        self.col_labels = col_labels
        self.matriz=None
        self.mapa_de_linhas = self.mapear_classes()

        #print(f"UUC{self.UUC_classes}")
        #print(f"target original{self.target_original}")
        #print(f"target test{self.target_test}")
        #print(f"predict{self.predict}")

    def set_data(self,predict,target_test,target_original):
        self.predict = predict+1
        self.target_test = target_test+1#eh preciso somar um pois podem haver targets = -1 no caso de usar um dataset inteiro como deconhecido junto com certas classes desconhecidas (como mnist + omniglot com omniglot e classes 7,8,9 como desconhecidas)
        self.target_original=target_original+1

    def mapear_classes(self):
        mapa_de_linhas = {}

        # Linha 0 reservada para "unknown" (classes fora de UUC_classes)
        linha_idx = 1  # Começa em 1 para as conhecidas

        for c in np.unique(self.target_original):
            #print(c)
            if c not in self.UUC_classes and c!=0:
                mapa_de_linhas[c] = linha_idx
                linha_idx += 1
            else:
                mapa_de_linhas[c] = 0
        #print(mapa_de_linhas)
        return mapa_de_linhas

    def computa_matriz(self):
        
        if(self.matriz is None):
            colunas = len(np.unique(self.target_original))
            linhas = len(np.unique(self.target_test))
            self.matriz=np.zeros((linhas,colunas)) # colunas: Omniglot, M0,M1,...,M9 -- targets reais
                                                #linhas: Unknown, classes conhecidas (MNIST - UUC) -- predicoes
        #print(self.mapa_de_linhas)
        
        for predict, target_original in zip(self.predict,self.target_original):
            predict = int(predict)
            target_original = int(target_original)
            linha = self.mapa_de_linhas[predict]
            coluna = target_original
            self.matriz[linha][coluna]+=1

        return self.matriz
    
    def exibe_matriz(self,dir=None,name=None):
        if self.matriz is None:
            print("Matriz não computada ainda.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # --- SOLUÇÃO: Valores > 0 em azul, Valor 0 em branco ---

        # 1. Copiamos um colormap existente. 'Blues' é ótimo para isso.
        cmap = plt.get_cmap('Blues').copy()

        # 2. Definimos a cor para valores "abaixo" do nosso intervalo.
        #    Como nosso intervalo começará em 1, o 0 será pintado de branco.
        cmap.set_under('white')
        # Esta é a abordagem correta para o seu objetivo
        cores = ["#FFFFFF", 'royalblue']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_lighter_blue', cores)

        # 3. Criamos uma normalização. O gradiente de cor será aplicado
        #    apenas a valores entre vmin e vmax.
        #    Qualquer valor < vmin (ou seja, 0) usará a cor de .set_under().
        norm = mcolors.Normalize(vmin=1, vmax=self.matriz.max())
        
        # 4. Usamos o cmap e a normalização personalizados no imshow.
        cax = ax.imshow(self.matriz, interpolation='nearest', cmap=cmap, norm=norm)
        fig.colorbar(cax)

        ax.set_title("Confusion Matrix", pad=20)

        # Eixo Y = previsão (linha 0 é "desconhecido")
        linhas_ordenadas = sorted(self.mapa_de_linhas.items(), key=lambda x: x[1])
        row_labels = ['Unknown'] + [str(classes) for idx,classes in enumerate(self.col_labels) if (idx!=0 and idx not in self.UUC_classes)]

        ax.set_xticks(np.arange(len(self.col_labels)))
        ax.set_xticklabels(self.col_labels, rotation=45, ha="left")

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        # Adiciona os números com cor de texto dinâmica (não mudou)
        threshold = self.matriz.max() / 2.
        
        for i in range(self.matriz.shape[0]):
            for j in range(self.matriz.shape[1]):
                valor = int(self.matriz[i, j])
                cor_texto = 'white' if self.matriz[i, j] > threshold else 'black'
                ax.text(j, i, str(valor), ha='center', va='center', color=cor_texto)

        ax.set_xlabel("Real Class")
        ax.set_ylabel("Predicted Class")
        plt.tight_layout()

        if dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(os.path.join(dir,f"Matriz de confusao_{name}.png"))
        else:
            plt.savefig(f"../../Matriz de confusao_{name}.png")

        plt.close()

class Matriz_confusao_osr_dataset_outlier:
    def __init__(self,predict,target_test,target_original,UUC_classes,col_labels):
        self.predict = predict+1
        self.target_test = target_test+1#eh preciso somar um pois podem haver targets = -1 no caso de usar um dataset inteiro como deconhecido junto com certas classes desconhecidas (como mnist + omniglot com omniglot e classes 7,8,9 como desconhecidas)
        self.target_original=target_original+1
        self.UUC_classes = np.array(UUC_classes)+1
        self.col_labels = col_labels
        self.matriz=None
        self.mapa_de_linhas = self.mapear_classes()

        print(f"UUC{self.UUC_classes}")
        print(f"target original{self.target_original}")
        print(f"target test{self.target_test}")
        print(f"predict{self.predict}")

    def mapear_classes(self):
        mapa_de_linhas = {}

        # Linha 0 reservada para "unknown" (classes fora de UUC_classes)
        linha_idx = 1  # Começa em 1 para as conhecidas

        for c in np.unique(self.target_original):
            print(c)
            if c not in self.UUC_classes and c!=0:
                mapa_de_linhas[c] = linha_idx
                linha_idx += 1
            else:
                mapa_de_linhas[c] = 0
        print(mapa_de_linhas)
        return mapa_de_linhas

    def computa_matriz(self):
        
        if(self.matriz==None):
            colunas = len(np.unique(self.target_original))
            linhas = len(np.unique(self.target_test))
            self.matriz=np.zeros((linhas,colunas)) # colunas: Omniglot, M0,M1,...,M9 -- targets reais
                                                #linhas: Unknown, classes conhecidas (MNIST - UUC) -- predicoes
        print(self.mapa_de_linhas)
        
        for predict, target_original in zip(self.predict,self.target_original):
            predict = int(predict)
            target_original = int(target_original)
            linha = self.mapa_de_linhas[predict]
            coluna = target_original
            self.matriz[linha][coluna]+=1
    
    def exibe_matriz(self,dir=None):
        if self.matriz is None:
            print("Matriz não computada ainda.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # --- SOLUÇÃO: Valores > 0 em azul, Valor 0 em branco ---

        # 1. Copiamos um colormap existente. 'Blues' é ótimo para isso.
        cmap = plt.get_cmap('Blues').copy()

        # 2. Definimos a cor para valores "abaixo" do nosso intervalo.
        #    Como nosso intervalo começará em 1, o 0 será pintado de branco.
        cmap.set_under('white')
        # Esta é a abordagem correta para o seu objetivo
        cores = ["#FFFFFF", 'royalblue']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_lighter_blue', cores)

        # 3. Criamos uma normalização. O gradiente de cor será aplicado
        #    apenas a valores entre vmin e vmax.
        #    Qualquer valor < vmin (ou seja, 0) usará a cor de .set_under().
        norm = mcolors.Normalize(vmin=1, vmax=self.matriz.max())
        
        # 4. Usamos o cmap e a normalização personalizados no imshow.
        cax = ax.imshow(self.matriz, interpolation='nearest', cmap=cmap, norm=norm)
        fig.colorbar(cax)

        ax.set_title("Confusion Matrix", pad=20)

        # Eixo Y = previsão (linha 0 é "desconhecido")
        linhas_ordenadas = sorted(self.mapa_de_linhas.items(), key=lambda x: x[1])
        row_labels = ['Unknown'] + [str(classes) for idx,classes in enumerate(self.col_labels) if (idx!=0 and idx not in self.UUC_classes)]

        ax.set_xticks(np.arange(len(self.col_labels)))
        ax.set_xticklabels(self.col_labels, rotation=45, ha="left")

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        # Adiciona os números com cor de texto dinâmica (não mudou)
        threshold = self.matriz.max() / 2.
        
        for i in range(self.matriz.shape[0]):
            for j in range(self.matriz.shape[1]):
                valor = int(self.matriz[i, j])
                cor_texto = 'white' if self.matriz[i, j] > threshold else 'black'
                ax.text(j, i, str(valor), ha='center', va='center', color=cor_texto)

        ax.set_xlabel("Real Class")
        ax.set_ylabel("Predicted Class")
        plt.tight_layout()

        if dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir+"Matriz de confusao.png")
        else:
            plt.savefig("../../Matriz de confusao.png")
        plt.show()
            
            



## ORIGINAL FUNCTIONS ------------------------------

# def find_anchor_means(net, mapping, datasetName, trial_num, cfg, only_correct = False):
#     ''' Tests data and fits a multivariate gaussian to each class' logits. 
#         If dataloaderFlip is not None, also test with flipped images. 
#         Returns means and covariances for each class. '''
#     #find gaussians for each class
#     if datasetName == 'MNIST' or datasetName == "SVHN":
#         loader, _ = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
#         logits, labels = gather_outputs(net, mapping, loader, only_correct = only_correct)
#     else:
#         loader, loaderFlipped = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
#         logits, labels = gather_outputs(net, mapping, loader, loaderFlipped, only_correct = only_correct)

#     num_classes = cfg['num_known_classes']
#     means = [None for i in range(num_classes)]

#     for cl in range(num_classes):
#         x = logits[labels == cl]
#         x = np.squeeze(x)
#         means[cl] = np.mean(x, axis = 0)

#     return means

# def gather_outputs(net, mapping, dataloader, dataloaderFlip = None, data_idx = 0, calculate_scores = False, unknown = False, only_correct = False):
    # ''' Tests data and returns outputs and their ground truth labels.
        # data_idx        0 returns logits, 1 returns distances to anchors
        # use_softmax     True to apply softmax
        # unknown         True if an unknown dataset
        # only_correct    True to filter for correct classifications as per logits
    # '''
    # X = []
    # y = []

    # if calculate_scores:
        # softmax = torch.nn.Softmax(dim = 1)

    # for i, data in enumerate(dataloader):
        # images, labels = data
        # images = images.cuda()

        # if unknown:
            # targets = labels
        # else:
            # targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()
        
        # outputs = net(images)
        # logits = outputs[0]
        # distances = outputs[1]

        # if only_correct:
            # if data_idx == 0:
                # _, predicted = torch.max(logits, 1)
            # else:
                # _, predicted = torch.min(distances, 1)
            
            # mask = predicted == targets
            # logits = logits[mask]
            # distances = distances[mask]
            # targets = targets[mask]

        # if calculate_scores:
            # softmin = softmax(-distances)
            # invScores = 1-softmin
            # scores = distances*invScores
        # else:
            # if data_idx == 0:
                # scores = logits
            # if data_idx == 1:
                # scores = distances

        # X += scores.cpu().detach().tolist()
        # y += targets.cpu().tolist()

    # if dataloaderFlip is not None:
        # for i, data in enumerate(dataloaderFlip):
            # images, labels = data
            # images = images.cuda()

            # if unknown:
                # targets = labels
            # else:
                # targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()
            
            # outputs = net(images)
            # logits = outputs[0]
            # distances = outputs[1]

            # if only_correct:
                # if data_idx == 0:
                    # _, predicted = torch.max(logits, 1)
                # else:
                    # _, predicted = torch.min(distances, 1)
                # mask = predicted == targets
                # logits = logits[mask]
                # distances = distances[mask]
                # targets = targets[mask]
                
            # if calculate_scores:
                # softmin = softmax(-distances)
                # invScores = 1-softmin
                # scores = distances*invScores
            # else:
                # if data_idx == 0:
                    # scores = logits
                # if data_idx == 1:
                    # scores = distances

            # X += scores.cpu().detach().tolist()
            # y += targets.cpu().tolist()

    # X = np.asarray(X)
    # y = np.asarray(y)

    # return X, y

