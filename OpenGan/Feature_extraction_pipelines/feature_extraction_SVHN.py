from Feat_extraction.ResNet18_32x32_feature_extraction import ResNet18_32x32_feature_extraction
import numpy as np
import torch
import os
from Utils import fix_random_seed, NOMES, ToUnknown
from Datasets import SVHNLoader
device = 'cuda:0'
seed = 42
fix_random_seed(seed)

def main():
    # 1. Configuração do Modelo
    num_classes = 10
    model = ResNet18_32x32_feature_extraction(num_classes=num_classes)
    model.load_model(torch.load(NOMES.RESNET18_CIFAR10.value))
    model.to(device)
    model.eval()

    # 2. Configuração de Caminhos
    features_dir = os.path.join(NOMES.FEATS_DIR.value, "SVHN", NOMES.RESNET18.value)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir, exist_ok=True)

    # 3. Inicialização do Loader e Transformações
    data_manager = SVHNLoader(root="./data/", download=True)
    transform = NOMES.CIFAR_RESNET18_VAL_TEST_TRANSFORMS.value 
    bs = 256

    # 4. Dicionário para iterar sobre os conjuntos
    loaders = {
        "train": data_manager.get_train_loader(transform=transform, bs=bs,target_transform=ToUnknown()),
        "val":   data_manager.get_val_loader(transform=transform, bs=bs,target_transform=ToUnknown()),
        "test":  data_manager.get_test_loader(transform=transform, bs=bs,target_transform=ToUnknown()),
    }

    # 5. Extração e Salvamento em Loop
    for split_name, loader in loaders.items():
        print(f"Extraindo características: SVHN - Conjunto de {split_name}...")

        suffix = f"SVHN_{split_name}"

        with torch.no_grad():
            model.save_features(loader, features_dir, suffix)

    print(f"\nTodos os conjuntos foram salvos em: {features_dir}")

if __name__ == "__main__":
    main()