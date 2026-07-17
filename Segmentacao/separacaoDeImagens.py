from PIL import Image
import os
import numpy as np
import glob
from multiprocessing import Pool, cpu_count

COLOR2CLASS = {
    (128,   0,   0): 1,  # Grass
    (0,   128,   0): 2,  # Ground
    (128, 128,   0): 0,  # Panicum
}

dir_mascara = "/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Mascaras/"
dir_original = "/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Imagens/"
dir_saida = "/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/"

teste = "/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Teste/"
treino = "/home/alexandreselani/Desktop/Segmentacao/ImagensCortadas/Alexandre/Dataset/Treino/"
os.makedirs(dir_saida, exist_ok=True)

mask_files = sorted(glob.glob(os.path.join(dir_mascara, "*.png")))
mask_files.extend(sorted(glob.glob(os.path.join(dir_mascara, "*.jpg"))))
original_files = sorted(glob.glob(os.path.join(dir_original, "*.png")))
original_files.extend(sorted(glob.glob(os.path.join(dir_original, "*.jpg"))))

dataset = glob.glob(os.path.join(teste, "*.png"))
dataset.extend(glob.glob(os.path.join(treino, "*.png")))

if not mask_files:
    print("Nenhuma máscara encontrada em", dir_mascara)
    exit(1)

def SeparaClasses(pair):
    mask_path, orig_path = pair

    imagem_mascara = np.array(Image.open(mask_path).convert("RGB"))
    imagem_original = np.array(Image.open(orig_path).convert("RGB"))

    if imagem_mascara.shape != imagem_original.shape:
        print(f"Tamanho incompatível entre {mask_path} e {orig_path}")
        return
 
    imagens = {
        0: np.zeros_like(imagem_original),
        1: np.zeros_like(imagem_original),
        2: np.zeros_like(imagem_original)
    }

    
    for rgb, cls in COLOR2CLASS.items():
        mask = np.all(imagem_mascara == rgb, axis=-1)
        if np.any(mask):
            imagens[cls][mask] = imagem_original[mask]

    base_name = os.path.splitext(os.path.basename(orig_path))[0]

    for cls, img in imagens.items():
        if np.any(img):
            out_path = os.path.join(dir_saida, f"{cls}_{base_name}.png")
            Image.fromarray(img).save(out_path)
            print(f"wrote {out_path}")

# Cria pares (máscara, imagem original)
pairs = list(zip(mask_files, original_files))

def checaTamanho():
    for path in dataset:
        img = np.array(Image.open(path).convert("RGB"))
        
        if img.shape != (320,640,3):
            print(path)

checaTamanho()