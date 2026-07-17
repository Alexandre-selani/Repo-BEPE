import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from Utils import ToUnknown, NOMES
from torchvision.datasets import ImageFolder

TRAIN       = "train_idx"
KKC_TEST    = "kkc_test_idx"
KKC_VAL     = "kkc_val_idx"
UUC_TEST    = "uuc_test_idx"
UUC_VAL     = "uuc_val_idx"

class Eucalyptus_closedset_loader:
    def __init__(self, bs,dataset="dataset-1"):
        self.bs = bs
        # Carrega os splits salvos anteriormente
        self.splits = torch.load(f"/home/alexandreselani/Desktop/Eucalyptus/eucalyptus_closedset_{dataset}_splits.pt", weights_only=False)
        
        # Caminhos base
    
        self.root = f"/home/alexandreselani/Desktop/Eucalyptus/all_images/closed_set/{dataset}"
        
        
    def _get_dataset(self, root, transform, is_uuc=False):
        """Helper para instanciar o dataset com o transform específico"""
        target_transf = ToUnknown() if is_uuc else None
        return ImageFolder(root=root, transform=transform, target_transform=target_transf)

    def load_train(self, fold, transform):
        if transform is None: raise ValueError("Transform não pode ser None para treino.")
        
        FOLD = f"fold_{fold}"
        ds_base = self._get_dataset(self.root, transform)
        train_dataset = Subset(ds_base, self.splits[FOLD][TRAIN])
        
        return DataLoader(
            train_dataset, batch_size=self.bs, shuffle=True,
            num_workers=4, pin_memory=True,persistent_workers=True
        )

    
    def load_test(self, fold, transform):
        """Cenário Open-Set: KKC + UUC com o mesmo transform de teste"""
        FOLD = f"fold_{fold}"
        
        ds_kkc_base = self._get_dataset(self.root, transform)
    

        test_kkc = Subset(ds_kkc_base, self.splits[FOLD][KKC_TEST])


        return DataLoader(
            test_kkc, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    def load_val(self,fold,transform):
        FOLD = f"fold_{fold}"
        
        ds_kkc_base = self._get_dataset(self.root, transform)


        val_kkc = Subset(ds_kkc_base, self.splits[FOLD][KKC_VAL])


        return DataLoader(
            val_kkc, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

class Eucalyptus_openset_loader:
    def __init__(self, bs, dataset="dataset-1"):
        self.bs = bs
        # Carrega os splits salvos anteriormente
        self.splits = torch.load(f"/home/alexandreselani/Desktop/Eucalyptus/eucalyptus_openset_{dataset}_splits.pt", weights_only=False)
        
        # Caminhos base
    
        self.kkc_root = f"/home/alexandreselani/Desktop/Eucalyptus/all_images/open_set/{dataset}/KKC/"
        self.uuc_root = f"/home/alexandreselani/Desktop/Eucalyptus/all_images/open_set/{dataset}/UUC/"
        
    def _get_dataset(self, root, transform, is_uuc=False):
        """Helper para instanciar o dataset com o transform específico"""
        target_transf = ToUnknown() if is_uuc else None
        return ImageFolder(root=root, transform=transform, target_transform=target_transf)

    def load_train(self, fold, transform):
        if transform is None: raise ValueError("Transform não pode ser None para treino.")
        
        FOLD = f"fold_{fold}"
        ds_base = self._get_dataset(self.kkc_root, transform)
        train_dataset = Subset(ds_base, self.splits[FOLD][TRAIN])
        
        return DataLoader(
            train_dataset, batch_size=self.bs, shuffle=True,
            num_workers=4, pin_memory=True,persistent_workers=True
        )

    def load_kkc_val(self, fold, transform):
        FOLD = f"fold_{fold}"
        ds_base = self._get_dataset(self.kkc_root, transform)
        val_kkc_dataset = Subset(ds_base, self.splits[FOLD][KKC_VAL])
        
        return DataLoader(
            val_kkc_dataset, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    def load_kkc_test(self, fold, transform):
        FOLD = f"fold_{fold}"
        ds_base = self._get_dataset(self.kkc_root, transform)
        kkc_test_ds = Subset(ds_base, self.splits[FOLD][KKC_TEST])
        
        return DataLoader(
            kkc_test_ds, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    def load_uuc_test(self, fold, transform):
        FOLD = f"fold_{fold}"
        ds_base = self._get_dataset(self.uuc_root, transform, is_uuc=True)
        uuc_test_ds = Subset(ds_base, self.splits[FOLD][UUC_TEST])
        
        return DataLoader(
            uuc_test_ds, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
    
    def load_uuc_val(self,fold,transform):
        FOLD = f"fold_{fold}"
        ds_base = self._get_dataset(self.uuc_root, transform, is_uuc=True)
        uuc_val_ds = Subset(ds_base, self.splits[FOLD][UUC_VAL])
        
        return DataLoader(
            uuc_val_ds, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
    
    def load_test(self, fold, transform):
        """Cenário Open-Set: KKC + UUC com o mesmo transform de teste"""
        FOLD = f"fold_{fold}"
        
        ds_kkc_base = self._get_dataset(self.kkc_root, transform)
        ds_uuc_base = self._get_dataset(self.uuc_root, transform, is_uuc=True)

        test_kkc = Subset(ds_kkc_base, self.splits[FOLD][KKC_TEST])
        test_uuc = Subset(ds_uuc_base, self.splits[FOLD][UUC_TEST])

        ds_concat = ConcatDataset([test_kkc, test_uuc])

        return DataLoader(
            ds_concat, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

    def load_val(self,fold,transform):
        FOLD = f"fold_{fold}"
        
        ds_kkc_base = self._get_dataset(self.kkc_root, transform)
        ds_uuc_base = self._get_dataset(self.uuc_root, transform, is_uuc=True)

        val_kkc = Subset(ds_kkc_base, self.splits[FOLD][KKC_VAL])
        val_uuc = Subset(ds_uuc_base, self.splits[FOLD][UUC_VAL])

        ds_concat = ConcatDataset([val_kkc, val_uuc])

        return DataLoader(
            ds_concat, batch_size=self.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )