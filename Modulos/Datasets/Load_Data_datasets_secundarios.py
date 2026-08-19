import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
# Nota: Certifique-se de que o módulo tinyimagenet esteja acessível
from tinyimagenet import TinyImageNet 

# class Tinyimagenet_loader:
#     def __init__(self, bs, transform=None, target_transform=None, root=None, download=False):
#         self.bs = bs
#         self.transform = transform
#         self.target_transform = target_transform
#         self.root = root if root is not None else "~/.torchvision/tinyimagenet/"
#         self.download = download

#     def load_train(self, transform=None, target_transform=None):
#         t = transform if transform is not None else self.transform
#         tt = target_transform if target_transform is not None else self.target_transform
#         train_set = TinyImageNet(root=self.root, split="train", transform=t, target_transform=tt)
#         return DataLoader(train_set, batch_size=self.bs, shuffle=True,
#                           num_workers=4, pin_memory=True, persistent_workers=True)

#     def load_val(self, transform=None, target_transform=None):
#         t = transform if transform is not None else self.transform
#         tt = target_transform if target_transform is not None else self.target_transform
#         val_set = TinyImageNet(root=self.root, split="val", transform=t, target_transform=tt)
#         return DataLoader(val_set, batch_size=self.bs, shuffle=False,
#                           num_workers=4, pin_memory=True, persistent_workers=True)

#     def load_test(self, transform=None, target_transform=None):
#         t = transform if transform is not None else self.transform
#         tt = target_transform if target_transform is not None else self.target_transform
#         test_set = TinyImageNet(root=self.root, split="test", transform=t, target_transform=tt)
#         return DataLoader(test_set, batch_size=self.bs, shuffle=False,
#                           num_workers=4, pin_memory=True, persistent_workers=True)


class CIFAR10Loader:
    def __init__(self, root='./data', download=False, target_transform=None):
        self.root = root
        self.download = download
        self.target_transform = target_transform

    def _get_train_val_datasets(self, transform, target_transform=None):
        tt = target_transform if target_transform is not None else self.target_transform
        full_train_dataset = datasets.CIFAR10(root=self.root, train=True, download=self.download, 
                                              transform=transform, target_transform=tt)

        indices = list(range(len(full_train_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
        return train_subset, val_subset

    def get_train_loader(self, transform, bs, target_transform=None):
        train_subset, _ = self._get_train_val_datasets(transform, target_transform)
        return DataLoader(train_subset, batch_size=bs, shuffle=True)

    def get_val_loader(self, transform, bs, target_transform=None):
        _, val_subset = self._get_train_val_datasets(transform, target_transform)
        return DataLoader(val_subset, batch_size=bs, shuffle=False)

    def get_test_loader(self, transform, bs, target_transform=None):
        tt = target_transform if target_transform is not None else self.target_transform
        test_dataset = datasets.CIFAR10(root=self.root, train=False, download=self.download, 
                                        transform=transform, target_transform=tt)
        return DataLoader(test_dataset, batch_size=bs, shuffle=False)



class SVHNLoader:
    def __init__(self, root='./data', download=False, target_transform=None):
        self.root = root
        self.download = download
        self.target_transform = target_transform

    def _get_train_val_datasets(self, transform, target_transform=None, size=5000):
        tt = target_transform if target_transform is not None else self.target_transform
        full_train_dataset = datasets.SVHN(root=self.root, split='train', download=self.download, 
                                           transform=transform, target_transform=tt)

        indices = list(range(len(full_train_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=size/len(indices), random_state=42)

        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
        return train_subset, val_subset

    def get_train_loader(self, transform, bs, target_transform=None):
        train_subset, _ = self._get_train_val_datasets(transform, target_transform)
        return DataLoader(train_subset, batch_size=bs, shuffle=True)

    def get_val_loader(self, transform, bs, target_transform=None, size=5000):
        _, val_subset = self._get_train_val_datasets(transform, target_transform, size)
        return DataLoader(val_subset, batch_size=bs, shuffle=False)

    def get_test_loader(self, transform, bs, target_transform=None, size=10000):
        tt = target_transform if target_transform is not None else self.target_transform
        test_dataset = datasets.SVHN(root=self.root, split='test', download=self.download, 
                                     transform=transform, target_transform=tt)
        indices = list(range(len(test_dataset)))
        _, test_indices = train_test_split(indices, test_size=size/len(indices), random_state=42)
        test_dataset_final = Subset(test_dataset, test_indices)
        return DataLoader(test_dataset_final, batch_size=bs, shuffle=False)