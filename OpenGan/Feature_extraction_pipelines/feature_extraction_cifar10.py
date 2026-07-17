from Feat_extraction.ResNet18_32x32_feature_extraction import ResNet18_32x32_feature_extraction
import numpy as np
import torch
import os
from Utils import fix_random_seed, NOMES
from Datasets import CIFAR10Loader 

device = 'cuda:0'
seed = 42
fix_random_seed(seed)

def main():
    # 1. Configuração do Modelo (Backbone treinado no TinyImageNet)
    num_classes = 10 
    model = ResNet18_32x32_feature_extraction(num_classes=num_classes)
    model.load_model(torch.load(NOMES.RESNET18_CIFAR10.value))
    model.to(device)
    model.eval()

    # 2. Configuração de Caminhos
    features_dir = os.path.join(NOMES.FEATS_DIR.value, "CIFAR", NOMES.RESNET18.value)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir, exist_ok=True)

    # 3. Inicialização do Loader e Transformações
    data_manager = CIFAR10Loader(root="./data/",download=False)
    # Usamos a transformação de validação/teste para extração de features (sem augmentations)
    transform = NOMES.CIFAR_RESNET18_VAL_TEST_TRANSFORMS.value
    bs = 256

    # 4. Dicionário para iterar sobre os conjuntos
    loaders = {
        "train": data_manager.get_train_loader(transform=transform, bs=bs),
        "val": data_manager.get_val_loader(transform=transform, bs=bs),
        "test": data_manager.get_test_loader(transform=transform, bs=bs)
    }

    # 5. Extração e Salvamento em Loop
    for split_name, loader in loaders.items():
        print(f"Extraindo características: CIFAR10 - Conjunto de {split_name}...")
        
        # O sufixo garante que os arquivos não se sobrescrevam (ex: CIFAR_train, CIFAR_val)
        suffix = f"CIFAR_{split_name}"
        
        with torch.no_grad():
            model.save_features(loader, features_dir, suffix)
            
    print(f"\nTodos os conjuntos foram salvos em: {features_dir}")

if __name__ == "__main__":
    main()