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
    """gera um .pt contendo os splits para 5 folds do dataset eucalyptus. Estrategia open set"""

    dataset = "dataset-1"

    val_size=0.1
    f = open(f"/home/alexandreselani/Desktop/Eucalyptus/tamanhos_folds_{dataset}_openset.txt", 'w') 
    
    
    KKCimages = ImageFolder(root=f"/home/alexandreselani/Desktop/Eucalyptus/all_images/open_set/{dataset}/KKC/")
    UUCimages = ImageFolder(root=f"/home/alexandreselani/Desktop/Eucalyptus/all_images/open_set/{dataset}/UUC/",target_transform=ToUnknown())
    
    SKFold = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    splits = {}

    for i, ((train_idx, test_idx), (test_uuc,val_uuc)) in enumerate(zip(SKFold.split(X=KKCimages.samples, y=KKCimages.targets), SKFold.split(X=UUCimages.samples,y=UUCimages.targets))):

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size,
            stratify=[KKCimages.targets[v] for v in train_idx],
            random_state=42,
        )

        splits[f"fold_{i}"] = { "train_idx": train_idx,
                                "kkc_test_idx": test_idx,
                                "kkc_val_idx": val_idx,
                                "uuc_val_idx": val_uuc,
                                "uuc_test_idx": test_uuc
                                }

        f.write(f"--------------FOLD {i}----------------\n")
        f.write(f"TRAIN: {len(train_idx)}\n\n")
        f.write(f"VAL_KKC: {len(val_idx)}\n")
        f.write(f"VAL_UUC: {len(val_uuc)}   TOTAL: {len(val_idx)+len(val_uuc)}\n\n")
        f.write(f"TEST_KKC: {len(test_idx)}\n")
        f.write(f"TEST_UUC: {len(test_uuc)}   TOTAL: {len(test_idx)+len(test_uuc)}\n\n")

    torch.save(splits,f"/home/alexandreselani/Desktop/Eucalyptus/eucalyptus_openset_{dataset}_splits.pt")
    f.close()

if __name__ == "__main__":
    main()