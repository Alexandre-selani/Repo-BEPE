"""Avaliacao open-set do GFROR em MNIST+Omniglot com ResNet18.

Avalia o classificador produzido por mnist_omni_resnet18/train_openset.py,
pareado com o autoencoder de mnist_omni_resnet18/train_generator.py: o AE gera
x_hat, o classificador recebe (x, x_hat) em 6 canais e a cabeca de
classificacao da a ativacao usada tanto para o rotulo quanto para o escore de
rejeicao (amostra vira "desconhecida" quando max_act < epsilon).

O pre-processamento e o RESNET18_MNIST_OMNI_EVAL_TRANSFORMS do enum NOMES
(Grayscale(3) + Resize(128) + ToTensor()), o mesmo do treino sem augmentation:
o AE foi treinado nessa escala e reconstroi em [0, 1].

Conjuntos (ver Datasets/Load_Data.py):
  - Val  = load_gridsearch(): 6k MNIST (conhecidas) + 6k Omniglot (desconhecidas),
           usado para escolher o epsilon;
  - Test = load_test(): 10k MNIST + 10k Omniglot.
"""

import os

import numpy as np
import pandas
import torch
import torchvision.transforms as T
from tqdm import tqdm

from model.vanilla_ae import VanillaAE128
from Modelos import ResNet18_GFROR
from Datasets import Mnist_omni_loader
from Utils import NOMES, fix_random_seed, metricasImplementadasV2, metricLogger

fix_random_seed(42)
# ─── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
MODEL = NOMES.RESNET18.value
NUM_CLASSES = 10
IMAGE_SIZE = 128
N_FOLDS = 1                # MNIST+Omniglot tem um unico split de classes conhecidas

result_dir = f"/home/alexandreselani/Desktop/GFROR/results/mnist_omni_resnet18/{MODEL}"
generator_path = "/home/alexandreselani/Desktop/GFROR/ckpt/ae_mnist_omni_resnet18/Mnist_omni/ckpt.pth"
classifier_path = f"/home/alexandreselani/Desktop/GFROR/ckpt/openset_ae_mnist_omni_resnet18/{MODEL}/ckpt.pth"
os.makedirs(result_dir, exist_ok=True)

column_names = ["Unknown"] + [str(i) for i in range(NUM_CLASSES)]

test_transform = NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value
data_manager = Mnist_omni_loader(BATCH_SIZE, test_transform)

G = torch.load(generator_path, weights_only=False, map_location=DEVICE).to(DEVICE)
C = torch.load(classifier_path, weights_only=False, map_location=DEVICE).to(DEVICE)


def predict(dataloader):
    G.eval()
    C.eval()

    all_targets = []
    all_max_act = []
    all_idx = []
    all_prob_known = []
    all_prob_unknown = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_hat = G(x)
            concat_x = torch.cat((x, x_hat), dim=1)
            out = C(concat_x)[0]

            max_act, indices = torch.max(out, dim=-1)

            # A K+1-esima classe implicita (a classe aberta) vem de assumir uma
            # saida linear extra de valor constante 0.
            z = torch.exp(out).sum(dim=1)
            prob_known = z / (z + 1)
            prob_unknown = 1 - prob_known

            all_targets.append(y.cpu())
            all_max_act.append(max_act.cpu())
            all_idx.append(indices.cpu())
            all_prob_known.append(prob_known.cpu())
            all_prob_unknown.append(prob_unknown.cpu())

    all_targets = torch.cat(all_targets, dim=0)
    all_max_act = torch.cat(all_max_act, dim=0)
    all_idx = torch.cat(all_idx, dim=0)
    all_prob_known = torch.cat(all_prob_known, dim=0)
    all_prob_unknown = torch.cat(all_prob_unknown, dim=0)

    print(pandas.DataFrame(all_max_act.numpy(), columns=["max_act"]).describe())
    return all_max_act, all_idx, all_prob_known, all_prob_unknown, all_targets


def threshold(max_act, preds, epsilon):
    predict = torch.where(max_act < epsilon, -1, preds)
    return predict


def run(dataloader, subdir, csv_name, epsilons):
    out_dir = os.path.join(result_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    logger = metricLogger(epsilons, N_FOLDS, out_dir, mc_column_names=column_names)

    max_act, preds, known_score, unknown_score, labels = predict(dataloader)

    for epsilon in epsilons:
        predicts = threshold(max_act, preds, epsilon)

        metricas = metricasImplementadasV2(predicts, labels,
                                           outlier_scores=-unknown_score,
                                           metodo="opengan")
        metricas = metricas._metricas()

        logger.update(metricas, 0, epsilon)
        logger.update_mc(epsilon, predicts, labels, labels)

    logger.aggregate(csv_name)


def val(epsilons):
    run(data_manager.load_gridsearch(), "Val", "Val.csv", epsilons)


def test(epsilons):
    run(data_manager.load_test(), "Test", "Test.csv", epsilons)


thresholds = np.arange(0, 30, 0.5)

val(thresholds)
test(thresholds)
