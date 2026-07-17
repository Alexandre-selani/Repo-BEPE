import torch
import numpy as np
import random
def fix_random_seed(seed: int = 12345) -> None:
    """
    Set all random seeds.

    :param seed: seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 
    
    # Impede que o CuDNN procure o melhor algoritmo (introduz ruído)
    torch.backends.cudnn.benchmark = False

from torchvision.datasets import VisionDataset
from torch.utils.data import Subset,random_split
import torch
import random

def random_dataset(dataset: VisionDataset, novo_tamanho: int):
    """
    Funcao que recebe um dataset e retorna um subconjunto de dados de tamanho novo_tamanho. os dados sao escolhidos aleatoriamente
    
    dataset: VisionDataset - dataset a ser reduzido
    novo_tamanho:int - numero de amostras que o subconjunto de dataset terá"""
    tamanho_antigo = len(dataset)
    assert novo_tamanho <= tamanho_antigo

    indices = random.sample(range(len(dataset)), novo_tamanho)

    
    subset = Subset(dataset,indices)
    return subset

def validation_split(porcentagem:float, dataset:VisionDataset):
    assert porcentagem > 0
    """Funcao que divide um conjunto de treino em dois subconjuntos disjuntos: de treino (novo) e de validacao
    
    porcentagem: float - porcentagem do dataset original a ser utilizada para validacao
    dataset: VisionDataset - dataset a ser dividido
    """
    validation_size = int(len(dataset)*porcentagem)
    train_size = len(dataset)-validation_size

    generator = torch.Generator()

    
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    return train_dataset,val_dataset

