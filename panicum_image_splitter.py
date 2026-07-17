"""Objective: split images in half, excluding images which have more black pixels than a set percentage

-----------------STEPS---------------------
1 - Load images from a class
2 - split each sample in half
3 - FOR EACH SAMPLE - check black allowance
4 - Repeat for the next class
"""

import os
import numpy as np
from PIL import Image
from typing import Any
from pathlib import Path
import shutil


def limpar_pathlib(caminho_pasta):
    pasta = Path(caminho_pasta)
    for item in pasta.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

ORIGINAL_IMG_DIR = "/home/alexandreselani/Desktop/Panicum completo/Dataset/"

NEW_IMG_DIR = "/home/alexandreselani/Desktop/Panicum_halfsize/"
os.makedirs(NEW_IMG_DIR,exist_ok=True)
limpar_pathlib(NEW_IMG_DIR)

ORIGINAL_HEIGHT = 320
ORIGINAL_WIDTH = 640


BLACK_ALLOWANCE = 0.97

CLASS2DIR = {"0": "UUC/2_Panicum",
             "1": "KKC/1_Milho",
             "2": "KKC/0_Solo"}



def split_sample(image):
    """Crops image into 2 subimages of half of the original size
    
    ------Parameters--------
    image:ImageFile -> image to be splitted

    ----------Returns--------
    (img1,img2):ImageFile -> resulting sub images
    """

    crop_area1 = (0,0,ORIGINAL_WIDTH//2,ORIGINAL_HEIGHT)
    crop_area2 = (ORIGINAL_WIDTH//2,0,ORIGINAL_WIDTH,ORIGINAL_HEIGHT)

    img1 = image.crop(crop_area1)
    img2 = image.crop(crop_area2)
    
    
    return img1,img2

def check_black_allowance(images):
    """
        Checks if the percentage of black pixels in each image is within the allowance.
        
        ------Parameters--------
        images: list[Image.Image] -> list containing the two halves (img1, img2)

        ----------Returns--------
        valid_images: list[Image.Image] -> list containing only images that passed the test
    """
    valid_images = []
    
    for img in images:
        # 1. Converter para numpy array para processamento rápido
        # Imagens RGB têm formato (H, W, 3)
        img_array = np.array(img)
        
        # 2. Definir o que é um pixel preto
        # Em imagens RGB, o preto absoluto é [0, 0, 0]
        # np.all(..., axis=-1) verifica se R, G e B são todos zero
        black_pixels_mask = np.all(img_array == 0, axis=-1)
        
        # 3. Calcular a quantidade e a porcentagem
        num_black_pixels = np.sum(black_pixels_mask)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        black_percentage = num_black_pixels / total_pixels
        
        # 4. Filtrar baseado no BLACK_ALLOWANCE (0.05)
        if black_percentage <= BLACK_ALLOWANCE:
            valid_images.append(img)
        
    return valid_images

def correct_size(image):
    """There may be some images smaller than the intended Height and Width. In this case, it must be corrected
    
    ------Parameters-------
    image:ImageFile -> sample to be corrected

    -----Returns-------
    new_img:ImageFile -> image with the correct size
    """

    new_img = Image.new('RGB', (ORIGINAL_WIDTH,ORIGINAL_HEIGHT), color='black')
    new_img.paste(image,(0,0))

    return new_img

def load_samples(dir) -> tuple[list[Any], str]:
    """Load all samples from dir and correct the size of eventual samples that are smaller than intended 
    
    ------Parameters-------
    dir:str -> class directory

    -----Returns-------
    loaded_samples: list[ImageFile] -> loaded samples
    klass:str -> class to which the samples belong
    """
    class_path = os.path.join(ORIGINAL_IMG_DIR,dir)
    samples = [f for f in os.listdir(class_path)]
    
    klass = os.path.basename(samples[0])
    klass = klass[0]

    print(f"Carregadas {len(samples)} da classe {CLASS2DIR[klass]}")
    loaded_samples = []

    #print(samples)
    for sample in samples:
        sample_path = os.path.join(class_path,sample)
        loaded_sample = Image.open(sample_path)

        if loaded_sample.size != (ORIGINAL_WIDTH,ORIGINAL_HEIGHT):
            loaded_sample = correct_size(loaded_sample)

        loaded_samples.append(loaded_sample)
    
    return loaded_samples,klass

def save_images(base_dir, images, klass, original_filename):
    """
    images: lista de imagens (metades) que passaram no teste de black_allowance
    original_filename: nome do arquivo original (ex: 'foto_01.jpg')
    """
    save_dir = os.path.join(base_dir, CLASS2DIR[klass])
    os.makedirs(save_dir, exist_ok=True)
    
    # Remove a extensão original para criar o prefixo
    file_prefix = Path(original_filename).stem
    
    for i, img in enumerate(images):
        # O nome do arquivo salvo conterá o nome da original
        save_path = os.path.join(save_dir, f"{file_prefix}_part{i}.png")
        img.save(save_path)

def main():
    class_folders = os.listdir(ORIGINAL_IMG_DIR)
    for folder in class_folders:
        # Caminho completo da pasta da classe
        class_path = os.path.join(ORIGINAL_IMG_DIR, folder)
        samples = os.listdir(class_path)
        
        if not samples: continue
        
        for sample_name in samples:
            sample_path = os.path.join(class_path, sample_name)
            
            # Carrega e corrige tamanho
            img = Image.open(sample_path)
            if img.size != (ORIGINAL_WIDTH, ORIGINAL_HEIGHT):
                img = correct_size(img)
            
            # Identifica a classe para a função de salvamento
            klass = sample_name[0] 

            # Divide e filtra
            img1, img2 = split_sample(img)
            resulting_crops = check_black_allowance([img1, img2])

            # Salva passando o nome original como referência
            if resulting_crops:
                save_images(NEW_IMG_DIR, resulting_crops, klass, sample_name)

            
    
if __name__ == "__main__":
    main()