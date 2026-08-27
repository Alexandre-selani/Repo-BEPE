"""Treino do classificador open-set do GFROR em MNIST+Omniglot com ResNet18.

Segunda etapa do GFROR: o autoencoder ja treinado (o gerador, ver
mnist_omni_resnet18/train_generator.py) fica congelado e produz x_hat; o
classificador recebe a concatenacao (x, x_hat) em 6 canais e e treinado com
duas cabecas:

  - classificacao das 10 classes conhecidas do MNIST (cross-entropy);
  - auto-supervisao: prever qual das 8 transformacoes deterministicas (rotacoes
    e flip+rotacao) foi aplicada ao par.

A perda total e 0.8 * ce + 0.2 * ss, como em mnist_omni_lenet/train_openset.py.

Diferencas em relacao ao pipeline com LeNet: o classificador e o
ResNet18_GFROR (stem 7x7/stride2 + maxpool, adequado a entradas 128x128) e as
imagens seguem as transformacoes RESNET18_MNIST_OMNI_* do enum NOMES, as mesmas
usadas no treino do autoencoder — mudar o pre-processamento aqui invalidaria os
x_hat que ele gera.
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from Datasets import Mnist_omni_loader
from Modelos import ResNet18_GFROR
from Utils import NOMES, fix_random_seed

fix_random_seed(42)

AE_PATH = "/home/alexandreselani/Desktop/GFROR/ckpt/ae_mnist_omni_resnet18/Mnist_omni/ckpt.pth"
SAVE_DIR = "/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_mnist_omni_resnet18"


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
    """Validacao closed-set: so as conhecidas (MNIST), so a cabeca de classificacao.

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

    lr = 0.001
    epochs = 30
    bs = 128
    num_classes = 10          # classes conhecidas: os 10 digitos do MNIST

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = NOMES.RESNET18.value
    save_dir = os.path.join(SAVE_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Mesmo pipeline do train_generator: transformacoes do enum NOMES (128x128, [0,1]).
    val_transform = NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value

    data_manager = Mnist_omni_loader(bs, val_transform) 

    train_dataloader = data_manager.load_train()
    val_kkc_dataloader = data_manager.load_mnist_val()

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

    classifier = ResNet18_GFROR(
        num_classes=num_classes,
        num_transforms=len(transformations),
        weights=None,
    ).to(device)

    # Gerador congelado: so gera x_hat.
    generator = torch.load(AE_PATH, weights_only=False, map_location=device).to(device)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.8)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):

        train_ce_loss, train_ss_loss, train_loss = train(
            generator, classifier, train_dataloader, optimizer, criterion,
            transformations, device)
        val_loss, val_acc = evaluate_closedSet(
            generator, classifier, val_kkc_dataloader, criterion, device)

        scheduler.step(val_loss)

        print('epoch [{}/{}], lr:{:.6f}, ce:{:.4f}, ss:{:.4f}, '
              'train loss:{:.4f}, val loss:{:.4f}, val acc:{:.4f}'.format(
                  epoch + 1, epochs, optimizer.param_groups[0]['lr'],
                  sum(train_ce_loss) / len(train_ce_loss),
                  sum(train_ss_loss) / len(train_ss_loss),
                  sum(train_loss) / len(train_loss), val_loss, val_acc), flush=True)

        # Seleciona pela acuracia closed-set nas conhecidas: e o unico criterio
        # legitimo aqui, ja que as desconhecidas nao podem ser usadas no treino.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(classifier.state_dict())
            print("    melhor modelo salvo na RAM (val acc {:.4f})".format(val_acc), flush=True)

    if best_state is not None:
        classifier.load_state_dict(best_state)
    torch.save(classifier, os.path.join(save_dir, "ckpt.pth"))
    print(f"classificador gravado em disco (melhor val acc {best_val_acc:.4f})", flush=True)


if __name__ == "__main__":
    main()
