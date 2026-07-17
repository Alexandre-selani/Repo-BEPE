import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.colors as mcolors
import os.path

class Matriz_confusao_osr_dataset_outlier:
    def __init__(self,predict,target_test,target_original,UUC_classes,col_labels):
        self.predict = predict
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

        ax.set_title("Matriz de Confusão OSR", pad=20)

        # Eixo Y = previsão (linha 0 é "desconhecido")
        linhas_ordenadas = sorted(self.mapa_de_linhas.items(), key=lambda x: x[1])
        row_labels = ['Desconhecido'] + [str(classes) for idx,classes in enumerate(self.col_labels) if (idx!=0 and idx not in self.UUC_classes)]

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

        ax.set_xlabel("Classe Real")
        ax.set_ylabel("Classe Prevista")
        plt.tight_layout()

        if dir:
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir+"Matriz de confusao.png")
        else:
            plt.savefig("../../Matriz de confusao.png")
        plt.show()
            
            

