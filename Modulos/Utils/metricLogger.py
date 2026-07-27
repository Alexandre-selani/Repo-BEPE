import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Utils import metricasImplementadasV2,Matriz_confusao_osr_dataset_outlier_cumulativa as mc

class metricLogger:
    """Gerencia o registro e agregação de métricas de classificação para experimentos Open Set Recognition (OSR).

    Acumula métricas (acurácia, F1, AUROC, etc.) organizadas por epsilon/threshold e por fold,
    além de gerenciar matrizes de confusão acumuladas ao longo dos folds. No final, gera:
      - Um CSV consolidado com média e desvio padrão por epsilon.
      - CSVs individuais por fold.
      - Matrizes de confusão acumuladas para cada epsilon (opcional).

    Args:
        epsilons: Lista de valores de epsilon (thresholds) a serem avaliados.
        n_folds: Número de folds da validação cruzada.
        dir: Diretório onde os resultados (CSVs e matrizes) serão salvos.
        flag_mc: Se True, mantém e exporta matrizes de confusão acumuladas.
        mc_column_names: Nomes das colunas para a matriz de confusão.
    """

    def __init__(self,epsilons,n_folds,dir,flag_mc=True,mc_column_names = ["Panicum","Ground", "Healthy"]):
        self.epsilons = [round(e,2) for e in epsilons]
        self.flag_mc = flag_mc
        self.dir = dir
        self.n_folds = n_folds
        self.mc_column_names = mc_column_names

        self.METRIC_KEYS = ("F1 macro", "accuracy", "UUC Accuracy", "inner metric", "outer metric", "halfpoint", "auroc")

        self.results_by_epsilon = {e:{key:[] for key in self.METRIC_KEYS} for e in self.epsilons}
        self.results_by_fold = {fold:[] for fold in range(n_folds)}
            
        if flag_mc:
            self.matrizes_confusao_acumulada = {e: None for e in self.epsilons}

    def update(self,metrics,fold,epsilon):
        epsilon = round(epsilon,2)
        """Registra as métricas de uma execução (combinação fold + epsilon).

        Args:
            metrics: Dicionário com as métricas calculadas (chaves em METRIC_KEYS).
            fold: Índice do fold atual.
            epsilon: Valor do epsilon utilizado nesta execução.
        """
        current_epsilon_fold_data = {"epsilon":epsilon}

        for metric in self.METRIC_KEYS:
            #update results_by_epsilon
            self.results_by_epsilon[epsilon][metric].append(metrics[metric])

            #update results_by_fold
            current_epsilon_fold_data[metric]=metrics[metric]

        self.results_by_fold[fold].append(current_epsilon_fold_data)

    def update_mc(self,epsilon,predicts,targets,original_targets):
        epsilon = round(epsilon,2)
        """Atualiza a matriz de confusão acumulada para um dado epsilon.

        Se a matriz ainda não existe para este epsilon, cria uma nova.
        Caso contrário, atualiza a existente com os novos dados.

        Args:
            epsilon: Valor do epsilon associado a estas predições.
            predicts: Lista/array de predições do classificador.
            targets: Lista/array de rótulos alvo (com -1 para desconhecidas).
            original_targets: Lista/array de rótulos originais (sem ajuste).
        """
        if self.matrizes_confusao_acumulada[epsilon] is None:
            matriz = mc(predicts, targets, original_targets, [], self.mc_column_names)
            matriz.computa_matriz()
            self.matrizes_confusao_acumulada[epsilon] = matriz
        else:
            self.matrizes_confusao_acumulada[epsilon].set_data(predicts, targets, original_targets)
            self.matrizes_confusao_acumulada[epsilon].computa_matriz()

    def aggregate(self,csv_name):
        """Consolida todos os resultados e gera os arquivos de saída.

        Para cada epsilon, calcula média e desvio padrão de cada métrica
        e exporta:
          - Um CSV consolidado (média ± std por epsilon).
          - CSVs individuais para cada fold (na subpasta "Folds").
          - Imagens das matrizes de confusão acumuladas (se flag_mc=True).

        Args:
            csv_name: Nome do arquivo CSV consolidado a ser gerado.
        """
        final_data = []
        os.makedirs(self.dir, exist_ok=True)

        for epsilon in sorted(self.results_by_epsilon.keys()):
            
            metrics = self.results_by_epsilon[epsilon]
            row = {
                "epsilon": epsilon,
            }
            
            for metric in self.METRIC_KEYS:
                row[f"{metric}_mean"] = np.mean(metrics[metric])
                row[f"{metric}_std"] = np.std(metrics[metric])

            final_data.append(row)
            
            if self.flag_mc:
                self.matrizes_confusao_acumulada[epsilon].exibe_matriz(dir=os.path.join(self.dir,"matrizes"), name=f"epsilon_{epsilon}")

        # ==========================================
        # 3. NOVO: Salva os arquivos de cada fold para o atual Alpha/Epsilon
        # ==========================================
        for fold in range(5):
            df_fold = pd.DataFrame(self.results_by_fold[fold])
            nome_arquivo_fold = f"Results_Fold_{fold}.csv"
            caminho_fold = os.path.join(self.dir,"Folds")
            os.makedirs(caminho_fold,exist_ok=True)
            caminho_fold = os.path.join(caminho_fold,nome_arquivo_fold)
            df_fold.to_csv(caminho_fold, index=False, float_format="%.3f")
            print(f"[*] Arquivo do Fold {fold} gerado: {nome_arquivo_fold}")

        df = pd.DataFrame(final_data)
        csv_path = os.path.join(self.dir, csv_name)
        df.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"Arquivo salvo: {csv_path}")
        



