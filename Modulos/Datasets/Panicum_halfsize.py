from torch.utils.data import DataLoader,ConcatDataset,Subset
from torchvision.datasets import ImageFolder
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50,ResNet50_Weights
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold,KFold,train_test_split,GroupShuffleSplit
import pandas as pd
import os

from Utils import ToUnknown,fix_random_seed,random_dataset
seed = 42
fix_random_seed(seed)

def gerar_dataset_csv(diretorio_base):
    data = []
    for root, dirs, files in os.walk(diretorio_base):
        for file in files:
            if file.endswith(".png"):
                # Extrai a classe pelo nome da pasta ou prefixo
                classe = os.path.basename(root) 
                
                # O NOME DO GRUPO é tudo que vem antes de "_part"
                # Exemplo: "campo_01_part0.png" -> grupo: "campo_01"
                grupo_id = file.split('_part')[0]
                
                #print(grupo_id)
                data.append({
                    'caminho': os.path.join(root, file),
                    'label': classe,
                    'group_id': grupo_id
                })
    return pd.DataFrame(data)


def main():
    f = open("/home/alexandreselani/Desktop/Experimento_panicum/Indices_panicum_halfsize/tamanhos_folds.txt", 'w') 
    test_size = 0.15

    # 1. Carregar metadados e ordenar para alinhar com ImageFolder
    df = gerar_dataset_csv("/home/alexandreselani/Desktop/Panicum_halfsize/")
    df = df.sort_values('caminho').reset_index(drop=True)
    
    # 2. Separar DataFrames por Categoria
    df_kkc = df[df['caminho'].str.contains("/KKC/")].copy().reset_index(drop=True)
    df_uuc = df[df['caminho'].str.contains("/UUC/")].copy().reset_index(drop=True)

    # 5. Configurar StratifiedGroupKFold para o KKC
    SKFold = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    
    splits = {}

    SKFold_UUC = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
uuc_fold_iterator = SKFold_UUC.split(df_uuc['caminho'], df_uuc['label'], groups=df_uuc['group_id'])
    # Loop dos Folds do KKC
    for i, (train_idx, test_idx) in enumerate(SKFold.split(df_kkc['caminho'], df_kkc['label'], groups=df_kkc['group_id'])):
        
        # --- ALEATORIZAÇÃO DO UUC POR FOLD (Mantendo Grupos) ---
        # Mudando o random_state para cada fold para garantir que o split UUC seja diferente
        
        test_idx_uuc, val_idx_uuc = next(uuc_fold_iterator)
        

        print(test_idx_uuc,val_idx_uuc)
        
        # Split de Validação dentro do Treino KKC (também por grupo/estratificado)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_sub_idx, val_sub_idx = next(gss_val.split(train_idx, groups=df_kkc.iloc[train_idx]['group_id']))
        
        # Mapeando os índices de volta para o array original do fold
        final_train_idx = train_idx[train_sub_idx]
        final_val_idx = train_idx[val_sub_idx]

        splits[f"fold_{i}"] = {
            "train_idx": final_train_idx.tolist(),
            "kkc_test_idx": test_idx.tolist(),
            "kkc_val_idx": final_val_idx.tolist(),
            "uuc_test_idx": test_idx_uuc.tolist(), 
            "uuc_val_idx": val_idx_uuc.tolist()
        }

        # --- LOG DE TAMANHOS E BALANCEAMENTO ---
        f.write(f"--- FOLD {i} ---\n")
        f.write(f"train: {len(final_train_idx)}\nkkc_test: {len(test_idx)}\nkkc_val: {len(final_val_idx)}\n")
        f.write(f"uuc_test: {len(test_idx_uuc)}\nuuc_val: {len(val_idx_uuc)}\n\n")
        
        # Escrevendo o balanceamento real dos índices selecionados neste fold
        f.write("CLASS BALANCE (KKC Train):\n")
        f.write(f"{df_kkc.iloc[final_train_idx]['label'].value_counts().to_string()}\n")
        f.write("CLASS BALANCE (UUC Test/Val):\n")
        f.write(f"{df_uuc.iloc[test_idx_uuc]['label'].value_counts().to_string()}\n")
        f.write(f"{df_uuc.iloc[val_idx_uuc]['label'].value_counts().to_string()}\n")
        f.write("-" * 20 + "\n\n")

    torch.save(splits, "/home/alexandreselani/Desktop/Experimento_panicum/Indices_panicum_halfsize/panicum_splits.pt")
    f.close()
    print("Splits salvos com sucesso!")

if __name__ == "__main__":
    main()