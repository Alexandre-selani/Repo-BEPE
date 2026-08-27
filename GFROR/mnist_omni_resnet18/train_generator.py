"""Treino do autoencoder (gerador) do GFROR para MNIST+Omniglot com ResNet18.

Primeira etapa do GFROR: o AE e treinado so nas classes conhecidas (as 10
classes do MNIST) e depois congelado, para gerar x_hat na segunda etapa.

Diferenca em relacao a mnist_omni_lenet/train_generator.py: as imagens seguem
as transformacoes RESNET18_MNIST_OMNI_* do enum NOMES (Grayscale(3 canais) +
Resize(128) + ToTensor(), com RandomAffine so no treino), ou seja, tensores
3x128x128 em [0, 1]. Por isso o AE usado aqui e o VanillaAE128, cujo decoder
termina em Sigmoid.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
from tqdm import tqdm

from Datasets import Mnist_omni_loader
from Utils import NOMES, fix_random_seed

from model.vanilla_ae import VanillaAE128

fix_random_seed(42)

IMAGE_SIZE = 128
CKPT_DIR = "/home/alexandreselani/Desktop/GFROR/ckpt/ae_mnist_omni_resnet18/Mnist_omni"
OUT_DIR = "/home/alexandreselani/Desktop/GFROR/output/ae_mnist_omni_resnet18/Mnist_omni"


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_losses = []

    for batch, _ in tqdm(dataloader):
        batch = batch.float().to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses


def evaluate(model, dataloader, loss_fn, device):

    model.eval()
    eval_losses = []

    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.float().to(device)
            loss = loss_fn(model(batch), batch)
            eval_losses.append(loss.item())

    return sum(eval_losses) / len(eval_losses)


def save_recons(model, dataloader, path, device, n=20):
    """Salva um grid alternando original e reconstrucao das primeiras n imagens."""
    model.eval()
    imgs = next(iter(dataloader))[0][:n].to(device)
    with torch.no_grad():
        out = model(imgs)
    # Sem normalizacao no pipeline: originais e saida do Sigmoid ja estao em [0, 1].
    org = imgs.cpu().clamp(0, 1)
    recons = out.cpu().clamp(0, 1)
    merged = torch.stack((org, recons), dim=1).view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    save_image(merged, path, nrow=8)


def main():

    config = {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "betas": (0.5, 0.999),
        "epochs": 30,
        "latent_size": 256,
        "type": "train mnist+omni ae (resnet18 pipeline, 128x128)",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mesmas transformacoes usadas para treinar o ResNet18 fechado neste dataset:
    # augmentation so no treino, validacao sem augmentation.
    val_transform = NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value

    data_manager = Mnist_omni_loader(config["batch_size"], val_transform)

    # O AE ve apenas as classes conhecidas (MNIST); Omniglot fica so na avaliacao.
    train_loader = data_manager.load_train()
    val_loader = data_manager.load_mnist_val()

    model = VanillaAE128(config["latent_size"]).to(device)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 betas=config["betas"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3,
                                  threshold_mode="abs")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    best_loss = float("inf")
    best_model_state = None

    for i in range(config["epochs"]):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        print('epoch [{}/{}], lr={:.6f}, train loss:{:.6f}, val loss:{:.6f}'.format(
            i + 1, config["epochs"], optimizer.param_groups[0]["lr"],
            sum(train_loss) / len(train_loss), val_loss), flush=True)

        if (i + 1) % 5 == 0:
            save_recons(model, val_loader,
                        os.path.join(OUT_DIR, "vanilla_recons_epoch{}.png".format(i + 1)),
                        device)

        if val_loss < best_loss:
            best_model_state = copy.deepcopy(model.state_dict())
            best_loss = val_loss
            print("    melhor modelo salvo na RAM (val loss {:.6f})".format(val_loss), flush=True)

    # Grava o melhor modelo em disco depois que todas as epocas terminarem.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model, os.path.join(CKPT_DIR, "ckpt.pth"))
    print(f"melhor modelo gravado em disco (val loss {best_loss:.6f})", flush=True)


if __name__ == "__main__":
    main()
