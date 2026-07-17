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
    """gera um .pt contendo os splits para 5 folds do dataset eucalyptus. Estrategia closed set"""

    val_size=0.1

    weights = ResNet50_Weights.IMAGENET1K_V1
    dataset = "dataset-1"
    images = ImageFolder(root=f"/home/alexandreselani/Desktop/Eucalyptus/all_images/closed_set/{dataset}")
    f = open(f"/home/alexandreselani/Desktop/Eucalyptus/tamanhos_folds_{dataset}_closedset.txt", 'w') 
    
    SKFold = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    splits = {}

    for i, (train_idx, test_idx) in enumerate(SKFold.split(X=images.samples, y=images.targets)):

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size,
            stratify=[images.targets[v] for v in train_idx],
            random_state=42,
        )

        splits[f"fold_{i}"] = { "train_idx": train_idx,
                                "kkc_test_idx": test_idx,
                                "kkc_val_idx": val_idx
                                }

        f.write(f"--------------FOLD {i}----------------\n")
        f.write(f"TRAIN: {len(train_idx)}\n\n")
        f.write(f"VAL: {len(val_idx)}\n")
        f.write(f"TEST: {len(test_idx)}\n")

    torch.save(splits,f"/home/alexandreselani/Desktop/Eucalyptus/eucalyptus_closedset_{dataset}_splits.pt")
    f.close()

if __name__ == "__main__":
    main()