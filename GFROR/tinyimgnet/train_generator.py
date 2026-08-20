"""Treino do autoencoder (gerador) do GFROR para o TinyImageNet.

Um AE por split de classes conhecidas: treina nas imagens de treino das classes
conhecidas do split e valida nas imagens conhecidas de validacao.

As imagens entram apenas com Resize(64) + ToTensor(), sem augmentation e sem
normalizacao, tanto no treino quanto na validacao. Por isso as imagens ficam em
[0, 1] e o decoder (Decoder64) termina com Sigmoid.
"""

import copy
import gc
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Datasets import TinyImageNet_loader
from Utils import fix_random_seed

from model.vanilla_ae import VanillaAE64

fix_random_seed(42)

IMAGE_SIZE = 64
DATA_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"


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
        "batch_size": 256,
        "learning_rate": 1e-4,
        "betas": (0.5, 0.999),
        "epochs": 40,
        "latent_size": 200,
        "type": "train tinyimgnet ae",
    }

    N_SPLITS = 5

    # Sem augmentation e sem normalizacao: o mesmo pipeline no treino e na validacao.
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    data_manager = TinyImageNet_loader(
        data_dir=DATA_DIR,
        splits_dir=SPLITS_DIR,
        batch_size=config["batch_size"],
        image_size=IMAGE_SIZE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for split in range(N_SPLITS):
        gc.collect()
        torch.cuda.empty_cache()

        best_loss = float("inf")
        best_model_state = None

        # Validacao so com as classes conhecidas: o AE e treinado apenas nelas.
        train_loader = data_manager.get_train_loader(split, transform)
        val_loader = data_manager.get_val_known_loader(split, transform)

        model = VanillaAE64(config["latent_size"]).to(device)
        loss_fn = nn.MSELoss(reduction="mean").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                     betas=config["betas"], weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3,
                                      threshold_mode="abs")

        ckpt_path = os.path.join("./ckpt/ae_tinyimgnet", "Tinyimgnet")
        out_path = os.path.join("./output/ae_tinyimgnet", "Tinyimgnet", f"split_{split}")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)

        for i in range(config["epochs"]):
            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            val_loss = evaluate(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)

            print('split {} epoch [{}/{}], lr={:.6f} train loss:{:.4f}, val loss:{:.4f}'.format(
                split, i + 1, config["epochs"], optimizer.param_groups[0]["lr"],
                sum(train_loss) / len(train_loss), val_loss))

            if (i + 1) % 10 == 0:
                save_recons(model, val_loader,
                            os.path.join(out_path, "vanilla_recons_epoch{}.png".format(i + 1)),
                            device)

            if val_loss < best_loss:
                best_model_state = copy.deepcopy(model.state_dict())
                best_loss = val_loss
                print("melhor modelo salvo na RAM")

        # Grava o melhor modelo em disco depois que todas as epocas terminarem.
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model, os.path.join(ckpt_path, f"split_{split}.pth"))
            print(f"melhor modelo do split {split} gravado em disco (val loss {best_loss:.4f})")

        optimizer.zero_grad()
        del train_loader, val_loader, model, optimizer, loss_fn, scheduler


if __name__ == "__main__":
    main()
