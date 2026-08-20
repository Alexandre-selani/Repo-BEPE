"""Treino do classificador open-set do GFROR no TinyImageNet.

Segunda etapa do GFROR: o autoencoder ja treinado (o gerador, ver
tinyimgnet/train_generator.py) fica congelado e produz x_hat; o classificador
recebe a concatenacao (x, x_hat) em 6 canais e e treinado com duas cabecas:

  - classificacao das classes conhecidas (cross-entropy);
  - auto-supervisao: prever qual das 8 transformacoes deterministicas (rotacoes
    e flip+rotacao) foi aplicada ao par.

A perda total e 0.8 * ce + 0.2 * ss, como no pipeline original.

As imagens entram apenas com Resize(64) + ToTensor(), sem augmentation e sem
normalizacao, coerente com o treino do autoencoder: o AE foi treinado nessa
escala e reconstroi em [0, 1], entao mudar o pre-processamento aqui invalidaria
os x_hat que ele gera.
"""

import copy
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from Datasets import TinyImageNet_loader
from Modelos import ResNet18_tinyimgnet_GFROR
from Utils import fix_random_seed, NOMES

fix_random_seed(42)

IMAGE_SIZE = 64
DATA_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"
AE_DIR = "/home/alexandreselani/Desktop/GFROR/ckpt/ae_tinyimgnet/Tinyimgnet"
SAVE_DIR = "/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_tinyimgnet"


# train on known classes, both classification and self supervision
def train(G, C, dataloader, optimizer, loss_fn, transformations, device):
    """Uma epoca: gerador congelado, classificador treinando as duas cabecas."""
    C.train()
    G.eval()

    ce_losses, ss_losses, train_losses = [], [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x_hat = G(x)

        concat_x = torch.cat((x, x_hat), dim=1)
        ce_loss = loss_fn(C(concat_x)[0], y)

        # note: how to get rid of for loop
        # A mesma transformacao e aplicada a x e a x_hat, para o par continuar alinhado.
        trans_ind = torch.randint(len(transformations), (x.size(0),))
        rand_trans = transformations[trans_ind]
        t_x = torch.stack([t(x[i]) for i, t in enumerate(rand_trans)], dim=0)
        t_x_hat = torch.stack([t(x_hat[i]) for i, t in enumerate(rand_trans)], dim=0)

        concat_t = torch.cat((t_x, t_x_hat), dim=1)
        ss_loss = loss_fn(C(concat_t)[1], trans_ind.to(device))

        loss = 0.8 * ce_loss + 0.2 * ss_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ce_losses.append(ce_loss.item())
        ss_losses.append(ss_loss.item())
        train_losses.append(loss.item())

    return ce_losses, ss_losses, train_losses


@torch.no_grad()
def evaluate_closedSet(G, C, dataloader, loss_fn, device):
    """Validacao closed-set: so as conhecidas, so a cabeca de classificacao.

    Retorna (loss media, acuracia).
    """
    C.eval()
    G.eval()

    ce_losses = []
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        x_hat = G(x)
        concat_x = torch.cat((x, x_hat), dim=1)
        logits = C(concat_x)[0]

        ce_losses.append(loss_fn(logits, y).item())
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return sum(ce_losses) / len(ce_losses), correct / total


def main():

    N_SPLITS = 5

    lr = 0.001
    epochs = 100
    bs = 128
    num_classes = 20          # classes conhecidas por split (protocolo OSR do TinyImageNet)
    warmup_epochs = 5         # epocas com o backbone congelado, so as cabecas treinando

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = NOMES.RESNET18.value
    save_dir = os.path.join(SAVE_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Mesmo pipeline do train_generator: sem augmentation e sem normalizacao.
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])

    data_manager = TinyImageNet_loader(
        data_dir=DATA_DIR,
        splits_dir=SPLITS_DIR,
        batch_size=bs,
        image_size=IMAGE_SIZE,
    )

    # As 8 transformacoes da tarefa auto-supervisionada.
    transformations = np.array([
        T.RandomRotation(degrees=[90, 90]),      # deterministic rotation
        T.RandomRotation(degrees=[180, 180]),
        T.RandomRotation(degrees=[270, 270]),
        T.RandomRotation(degrees=[360, 360]),    # original input
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[90, 90])]),  # deterministic flip + rotation
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[180, 180])]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[270, 270])]),
        T.Compose([T.RandomHorizontalFlip(p=1.), T.RandomRotation(degrees=[360, 360])]),
    ])

    for split in range(N_SPLITS):
        gc.collect()
        torch.cuda.empty_cache()

        split_dir = os.path.join(save_dir, f"Split_{split}")
        os.makedirs(split_dir, exist_ok=True)

        classifier = ResNet18_tinyimgnet_GFROR(
            num_classes=num_classes,
            num_transforms=len(transformations),
            weights=None,
        ).to(device)

        # Gerador do split correspondente, congelado: so gera x_hat.
        ae_path = os.path.join(AE_DIR, f"split_{split}.pth")
        generator = torch.load(ae_path, weights_only=False, map_location=device).to(device)
        generator.eval()
        for param in generator.parameters():
            param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.8)

        train_dataloader = data_manager.get_train_loader(split, transform)
        val_kkc_dataloader = data_manager.get_val_known_loader(split, transform)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(epochs):

            # Warmup: primeiro so as cabecas lineares, depois a rede inteira.
            # if epoch == 0:
            #     for param in classifier.parameters():
            #         param.requires_grad = False
            #     for param in classifier.classification.parameters():
            #         param.requires_grad = True
            #     for param in classifier.transformation.parameters():
            #         param.requires_grad = True
            # elif epoch == warmup_epochs:
            #     for param in classifier.parameters():
            #         param.requires_grad = True

            train_ce_loss, train_ss_loss, train_loss = train(
                generator, classifier, train_dataloader, optimizer, criterion,
                transformations, device)
            val_loss, val_acc = evaluate_closedSet(
                generator, classifier, val_kkc_dataloader, criterion, device)

            scheduler.step(val_loss)

            print('split {} epoch [{}/{}], lr:{:.6f}, ce:{:.4f}, ss:{:.4f}, '
                  'train loss:{:.4f}, val loss:{:.4f}, val acc:{:.4f}'.format(
                      split, epoch + 1, epochs, optimizer.param_groups[0]['lr'],
                      sum(train_ce_loss) / len(train_ce_loss),
                      sum(train_ss_loss) / len(train_ss_loss),
                      sum(train_loss) / len(train_loss), val_loss, val_acc))

            # Seleciona pela acuracia closed-set nas conhecidas: e o unico criterio
            # legitimo aqui, ja que as desconhecidas nao podem ser usadas no treino.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(classifier.state_dict())
                print("    melhor modelo salvo na RAM (val acc {:.4f})".format(val_acc))

        if best_state is not None:
            classifier.load_state_dict(best_state)
        torch.save(classifier, os.path.join(split_dir, "ckpt.pth"))
        print(f"split {split} gravado em disco (melhor val acc {best_val_acc:.4f})")

        del classifier, generator, train_dataloader, val_kkc_dataloader
        del optimizer, criterion, scheduler


if __name__ == "__main__":
    main()
