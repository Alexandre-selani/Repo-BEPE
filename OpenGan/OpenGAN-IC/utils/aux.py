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