import torch
import random
import numpy as np
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset,random_split
from sklearn.model_selection import StratifiedKFold

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

class ToUnknown(object):
    """
    Callable that returns a negative number, used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1



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

def validation_split(porcentagem:float, dataset):
    assert porcentagem > 0
    """Funcao que divide um conjunto de treino em dois subconjuntos disjuntos: de treino (novo) e de validacao
    
    porcentagem: float - porcentagem do dataset original a ser utilizada para validacao
    dataset: VisionDataset - dataset a ser dividido
    """
    validation_size = int(len(dataset)*porcentagem)
    n_splits = int(len(dataset)/validation_size)

    k_fold = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=42)
    iterator = iter(k_fold.split(dataset,dataset.targets))
    train_idx,val_idx = next(iterator)
    
    train_subset = Subset(dataset,train_idx)
    val_subset = Subset(dataset,val_idx)
    return train_subset,val_subset


def CACLoss(distances, gt,num_classes,lbda):
	'''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
	true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
	non_gt = torch.Tensor([[i for i in range(num_classes) if gt[x] != i] for x in range(len(distances))]).long().cuda()
	others = torch.gather(distances, 1, non_gt)
	
	anchor = torch.mean(true)

	tuplet = torch.exp(-others+true.unsqueeze(1))
	tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim = 1)))

	total = lbda*anchor + tuplet

	return total, anchor, tuplet
