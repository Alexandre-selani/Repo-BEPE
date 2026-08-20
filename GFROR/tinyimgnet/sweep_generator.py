"""Varredura de latent_size x funcao de perda para o AE (gerador) do GFROR no TinyImageNet.

O que este script responde: qual combinacao produz o maior *gap* de reconstrucao
entre classes conhecidas e desconhecidas -- que e o sinal do qual o GFROR depende
(a diferenca x - x_hat alimenta o classificador). Uma val loss baixa nao e o
objetivo: um AE que reconstroi tudo bem, inclusive o desconhecido, e inutil aqui.

Metodologia:
  - As losses de treino (L1 e MSE) tem escalas diferentes e NAO sao comparaveis
    entre si. Por isso a avaliacao usa sempre a mesma metrica neutra, o erro
    quadratico medio por imagem, independentemente da loss usada no treino.
  - O numero reportado como decisao e o AUROC obtido usando esse erro como score
    de "desconhecido". E um limite inferior do que o GFROR completo alcanca (aqui
    nao ha classificador), mas serve para ordenar as configuracoes.
  - Roda por padrao num unico split, que basta para escolher a configuracao.

Uso:
    python -m tinyimgnet.sweep_generator
    python -m tinyimgnet.sweep_generator --latents 128 256 --losses l1 --epochs 20
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from Datasets import TinyImageNet_loader
from Utils import fix_random_seed

from model.vanilla_ae import VanillaAE64

IMAGE_SIZE = 64
DATA_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"

LOSSES = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
}


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for batch, _ in tqdm(dataloader, leave=False):
        batch = batch.float().to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def recon_errors(model, dataloader, device):
    """Erro quadratico medio POR IMAGEM.

    Metrica fixa, usada para todas as configuracoes: e o que torna treinos com
    L1 e com MSE comparaveis entre si.
    """
    model.eval()
    errors = []
    for batch, _ in tqdm(dataloader, leave=False):
        batch = batch.float().to(device)
        out = model(batch)
        per_image = ((out - batch) ** 2).mean(dim=(1, 2, 3))
        errors.append(per_image.cpu())
    return torch.cat(errors).numpy()


def run_config(latent_size, loss_name, split, epochs, batch_size, lr, data_manager,
               transform, device):
    """Treina uma configuracao e mede a separacao conhecido/desconhecido."""
    fix_random_seed(42)  # mesma inicializacao para todas as configuracoes

    train_loader = data_manager.get_train_loader(split, transform)
    val_known_loader = data_manager.get_val_known_loader(split, transform)
    val_unknown_loader = data_manager.get_val_unknown_loader(split, transform)

    model = VanillaAE64(latent_size).to(device)
    loss_fn = LOSSES[loss_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999),
                                 weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=3,
                                  threshold_mode="abs")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # O scheduler acompanha a metrica neutra nas conhecidas, nao a loss de treino.
        known_err = recon_errors(model, val_known_loader, device)
        scheduler.step(float(known_err.mean()))
        print("    epoca [{}/{}] train {} loss {:.5f} | mse/img conhecidas {:.5f}".format(
            epoch + 1, epochs, loss_name, train_loss, known_err.mean()))

    known_err = recon_errors(model, val_known_loader, device)
    unknown_err = recon_errors(model, val_unknown_loader, device)

    # AUROC usando o erro de reconstrucao como score de "desconhecido":
    # o desconhecido deveria reconstruir PIOR, entao erro maior => label 1.
    y_true = np.concatenate([np.zeros(len(known_err)), np.ones(len(unknown_err))])
    y_score = np.concatenate([known_err, unknown_err])
    auroc = roc_auc_score(y_true, y_score)

    result = {
        "latent_size": latent_size,
        "loss": loss_name,
        "known_mse": float(known_err.mean()),
        "unknown_mse": float(unknown_err.mean()),
        "gap": float(unknown_err.mean() - known_err.mean()),
        # Gap normalizado pela dispersao das conhecidas: um gap absoluto grande
        # nao ajuda se as conhecidas tambem variam muito.
        "gap_norm": float((unknown_err.mean() - known_err.mean()) / (known_err.std() + 1e-12)),
        "auroc": float(auroc),
    }

    del model, optimizer, loss_fn, scheduler, train_loader
    del val_known_loader, val_unknown_loader
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latents", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--losses", nargs="+", default=["l1", "mse"], choices=list(LOSSES))
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="./output/ae_tinyimgnet/sweep.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mesmo pipeline do train_generator: sem augmentation e sem normalizacao.
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    data_manager = TinyImageNet_loader(
        data_dir=DATA_DIR,
        splits_dir=SPLITS_DIR,
        batch_size=args.batch_size,
        image_size=IMAGE_SIZE,
        splits=[args.split],   # evita calcular norm stats dos splits nao usados
    )

    results = []
    for latent_size in args.latents:
        for loss_name in args.losses:
            print(f"\n=== latent {latent_size} | loss {loss_name} | split {args.split} ===")
            result = run_config(latent_size, loss_name, args.split, args.epochs,
                                args.batch_size, args.lr, data_manager, transform, device)
            results.append(result)
            print("    -> conhecidas {known_mse:.5f} | desconhecidas {unknown_mse:.5f} "
                  "| gap {gap:.5f} | gap_norm {gap_norm:.3f} | AUROC {auroc:.4f}".format(**result))

    print("\n" + "=" * 78)
    print("{:>8} {:>6} {:>12} {:>13} {:>10} {:>10} {:>8}".format(
        "latente", "loss", "mse_conhec", "mse_descon", "gap", "gap_norm", "AUROC"))
    print("-" * 78)
    for r in sorted(results, key=lambda r: -r["auroc"]):
        print("{latent_size:>8} {loss:>6} {known_mse:>12.5f} {unknown_mse:>13.5f} "
              "{gap:>10.5f} {gap_norm:>10.3f} {auroc:>8.4f}".format(**r))
    print("=" * 78)

    best = max(results, key=lambda r: r["auroc"])
    print("\nMelhor por AUROC: latente {latent_size}, loss {loss} (AUROC {auroc:.4f})".format(**best))
    print("Lembrete: e um proxy so-reconstrucao, sem o classificador. Serve para ordenar\n"
          "configuracoes, nao como resultado final do GFROR.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {args.out}")


if __name__ == "__main__":
    main()
