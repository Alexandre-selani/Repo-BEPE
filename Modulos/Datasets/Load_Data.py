import torch
from Utils import AnaliseGraficaVal, train, eval, fix_random_seed, ToUnknown
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision.models import alexnet
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, Omniglot
import torch.nn as nn
import torch.optim as optim
from Utils import NOMES
fix_random_seed(42)

class Mnist_omni_loader:
    """
    Gerencia o carregamento e divisão dos datasets MNIST e Omniglot.
    
    Quantidades de imagens:
    - train_dataloader: 54000 imagens mnist
    - val_mnist_dataloader: 6000 imagens mnist
    - val_omni_dataloader: 6000 imagens omniglot
    - grid_search_val_dataloader: 6000 imagens mnist + 6000 imagens omniglot
    - test_dataloader: 10000 imagens mnist + 10000 imagens omniglot

    Metodos principais:
    - load_train: Carrega subset de treino (MNIST).
    - load_mnist_val: Carrega subset de validacao (MNIST).
    - load_omni_val: Carrega subset de validacao (Omniglot).
    - load_gridsearch: Carrega conjunto misto para busca de hiperparametros.
    - load_mnist_test: Carrega o teste padrao do MNIST.
    - load_test: Carrega o teste final misto (MNIST + Omniglot).

    invert_colors (padrao True) inverte apenas o Omniglot, para que ambos os
    conjuntos fiquem com tracos claros sobre fundo escuro. Com False os dois
    conjuntos ficam com polaridades opostas e o problema se torna trivial.
    """

    def __init__(self, bs,transform,invert_omni_colors=True):
        self.bs = bs
        self.splits = torch.load(NOMES.MNIST_OMNI_SPLITS.value, weights_only=False)
        
        if transform is None:
            raise ValueError

        self.mnist_train = MNIST(root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data", download=True, transform=transform, train=True)
        self.mnist_test = MNIST(root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data", download=True, transform=transform, train=False)

        # O Omniglot vem com tracos pretos sobre fundo branco; o MNIST e o inverso.
        # Sem inverter, a intensidade media do pixel sozinha separa conhecido de
        # desconhecido perfeitamente (AUROC 1.0, zero sobreposicao), ou seja, o
        # benchmark pode ser resolvido sem olhar a forma do caractere. Inverter o
        # Omniglot remove esse atalho. Os transforms terminam em ToTensor(), entao
        # a inversao age sobre um tensor float em [0, 1].
        omniglot_transform = transform
        if invert_omni_colors:
            omniglot_transform = transforms.Compose([
                transform,
                transforms.RandomInvert(p=1.0),
            ])

        self.test_omniglot = Omniglot(
            root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data",
            download=True,
            transform=omniglot_transform,
            target_transform=ToUnknown()
        )

        self.val_omniglot = Omniglot(
            root="/home/alexandreselani/Desktop/Experimento_mnist_omni/data",
            download=True,
            transform=omniglot_transform,
            target_transform=ToUnknown(),
            background=False
        )

    def load_train(self):
        """Cria subset de treino do MNIST baseado nos indices de split e retorna o DataLoader."""
        train_dataset = Subset(self.mnist_train, self.splits["mnist_train_idx"])
        return DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    
    def load_mnist_val(self):
        """Cria subset de validacao puramente MNIST para monitoramento de overfitting."""
        mnist_val_dataset = Subset(self.mnist_train, self.splits["mnist_val_idx"])
        return DataLoader(mnist_val_dataset, batch_size=self.bs, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    
    def load_omni_val(self):
        """Cria subset de validacao puramente Omniglot para testar rejeicao de classe desconhecida."""
        val_omniglot_reduzido = Subset(self.val_omniglot, self.splits["omniglot_val_idx"])
        return DataLoader(val_omniglot_reduzido, batch_size=self.bs, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    def load_gridsearch(self):
        """Concatena validacao MNIST e Omniglot para otimizacao de limiares ou hiperparametros."""
        val_omniglot_reduzido = Subset(self.val_omniglot, self.splits["omniglot_val_idx"])
        mnist_val_dataset = Subset(self.mnist_train, self.splits["mnist_val_idx"])
        grid_search_val_dataset = ConcatDataset([mnist_val_dataset, val_omniglot_reduzido])
        return DataLoader(grid_search_val_dataset, batch_size=self.bs, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    
    def load_mnist_test(self):
        """Retorna o set de teste padrao do MNIST (10k imagens)."""
        return DataLoader(self.mnist_test, batch_size=self.bs, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    
    def load_test(self):
        """Concatena MNIST test e Omniglot test para avaliacao final do modelo em Out-of-Distribution."""
        test_omniglot_reduzido = Subset(self.test_omniglot, self.splits["omniglot_test_idx"])
        test_dataset = ConcatDataset([test_omniglot_reduzido, self.mnist_test])
        return DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    def load_omni_test(self):
        """Retorna o set de teste reduzido do OMNIGLOT (10k imagens)."""
        test_omniglot_reduzido = Subset(self.test_omniglot, self.splits["omniglot_test_idx"])
        
        return DataLoader(test_omniglot_reduzido, batch_size=self.bs, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)