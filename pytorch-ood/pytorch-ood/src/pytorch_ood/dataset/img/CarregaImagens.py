import os
from PIL import Image
from torchvision.datasets import VisionDataset
import numpy as np
import glob


class CarregaImagens(VisionDataset):
    """
    Classe destinada a carregar imagens de uma pasta em um dataset do tipo VisionDataset.
    Eh necessario para utilizar a torchosr.
    
    :root: str
    :transform: torchvision.transforms
    """
    def __init__(self, root, transform=None,target_transform=None):
        super(CarregaImagens,self).__init__(root,transform=transform, target_transform=target_transform)

        files = sorted(glob.glob(os.path.join(root, "*.png")))
        files.extend(sorted(glob.glob(os.path.join(root, "*.jpg"))))
        files.extend(sorted(glob.glob(os.path.join(root, "*.jpeg"))))

        self.imagens = []
        self.classes={}
        self.rotulos=[]


        for path in files:#para cada arquivo do dataset
            label = int(os.path.basename(path)[0]) #extrai o primeiro caracter do nome do arquivo (representa a classe)

           
            if path.endswith(('.jpg','.png','.jpeg')):#salva o caminho e o rotulo de todas as imagens existentes em caminhoDaClasse
                self.imagens.append(path)
                self.rotulos.append(label)
        
        self.imagens = np.array(self.imagens)
        self.rotulos = np.array(self.rotulos)
    def __getitem__(self, index):
        caminhoImagem = self.imagens[index]
        rotulo = self.rotulos[index]

        imagem=Image.open(caminhoImagem).convert("RGB") # pega uma imagem indexada por index, carrega ela e converte para RGB
        
        if self.transform:
            imagem = self.transform(imagem) #aplica transformacoes na imagem
        
        return imagem,rotulo
    
    def __len__(self):
        return len(self.imagens) #comprimento do conjunto de dados
    
    def _n_classes(self):
        return len(np.unique(self.rotulos))



if __name__ == "__main__":
    a = CarregaImagens("/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Treino")

    print(a.imagens)
    print(a.rotulos)
    print(a.__getitem__(5))



