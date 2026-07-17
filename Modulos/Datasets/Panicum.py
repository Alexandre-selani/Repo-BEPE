from torch.utils.data import DataLoader,ConcatDataset,Subset
from torchvision.datasets import ImageFolder
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50,ResNet50_Weights

from sklearn.model_selection import StratifiedKFold,KFold,train_test_split


from Utils import ToUnknown,fix_random_seed,random_dataset
seed = 42
fix_random_seed(seed)

def main():
    """gera um .pt contendo os splits para 5 folds do dataset panicum. O conjunto de testes é balanceado entre classes conhecidas e desconhecidas"""

    f = open("/home/alexandreselani/Desktop/Experimento_panicum/Indices_panicum/tamanhos_folds.txt", 'w') 
    test_size=0.1

    weights = ResNet50_Weights.IMAGENET1K_V1

    panicum_kkc = ImageFolder(root="/home/alexandreselani/Desktop/Experimento_panicum/data/KKC/")
    panicum_uuc = ImageFolder(root="/home/alexandreselani/Desktop/Experimento_panicum/data/UUC/",target_transform=ToUnknown())

    SKFold = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

    splits = {}

    for i, (train_idx, test_idx) in enumerate(SKFold.split(X=panicum_kkc.samples, y=panicum_kkc.targets)):

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=test_size,
            stratify=[panicum_kkc.targets[v] for v in train_idx],
            random_state=42,
        )

        
        
        
        
        splits[f"fold_{i}"] = { "train_idx": train_idx,
                                "kkc_test_idx": test_idx,
                                "kkc_val_idx": val_idx,
                                "uuc_test_idx": panicum_uuc.targets}

        
        f.write(f"tamanhos FOLD {i}\n")
        f.write(f"train: {len(train_idx)}\nkkc_test_idx:{len(test_idx)}\nkkc_val: {len(val_idx)}\nuuc_test: {len(panicum_uuc.targets)}\n\n")


    torch.save(splits,"/home/alexandreselani/Desktop/Experimento_panicum/Indices_panicum/panicum_splits.pt")
    f.close()

if __name__ == "__main__":
    main()