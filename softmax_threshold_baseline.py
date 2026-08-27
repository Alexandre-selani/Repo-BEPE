"""
Baseline ingenua de Open Set Recognition: threshold na softmax (MSP).

Nenhum metodo de deteccao de novidade e usado aqui: os backbones fechados ja treinados
sao carregados do disco e, para cada amostra, a maior probabilidade da softmax (Maximum
Softmax Probability) e comparada a um threshold:

    predicao = argmax(softmax(logits))   se   max(softmax(logits)) >= threshold
    predicao = -1 (desconhecida)         caso contrario

Protocolo (um threshold por experimento):
  1. Varredura de thresholds de 0 a 1 com passo 0.01 no conjunto de VALIDACAO,
     registrando todas as metricas com metricLogger/metricasImplementadasV2.
  2. Escolha do threshold com maior F1 macro (media dos folds/splits quando o
     experimento usa varias particoes).
  3. Aplicacao do melhor threshold no conjunto de TESTE, registrando as metricas
     e a matriz de confusao acumulada.

Experimentos (backbones lidos do disco):
  mnist_omni_resnet18  ResNet18  (10 classes)  - execucao unica
  mnist_omni_lenet     LeNet     (10 classes)  - execucao unica
  panicum              ResNet18  (2 classes)   - 5 folds
  eucalyptus_dataset-1  AlexNet  (2 classes)   - 5 folds (open set loader)
  eucalyptus_dataset-20 AlexNet  (2 classes)   - 5 folds (open set loader)
  tinyimgnet           ResNet18  (20 classes)  - 5 splits

Uso:
    python softmax_threshold_baseline.py                    # roda todos
    python softmax_threshold_baseline.py --experimentos panicum tinyimgnet
"""

import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from torchvision.models import AlexNet_Weights

from Utils import fix_random_seed, NOMES, metricasImplementadasV2, metricLogger
from Datasets import (Mnist_omni_loader, Panicum_halfsize_loader,
                      Eucalyptus_openset_loader, TinyImageNet_loader)
from Modelos import ResNet18, LeNet, Alexnet, ResNet18_tinyimgnet

seed = 42
fix_random_seed(seed)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Thresholds de 0 a 1, passo 0.01 (101 valores). O metricLogger arredonda para 2 casas.
THRESHOLDS = [round(float(t), 3) for t in np.arange(0, 1.0 + 1e-9, 0.01)]

OUTPUT_ROOT = "/home/alexandreselani/Desktop/Resultados_softmax_threshold"

# Mesmas chaves devolvidas por metricasImplementadasV2._metricas / registradas pelo metricLogger.
METRIC_KEYS = ("F1 macro", "accuracy", "UUC Accuracy", "inner metric",
               "outer metric", "halfpoint", "auroc")

# ── Backbones ja treinados (pesos em disco) ───────────────────────────────────
MNIST_OMNI_RESNET18 = "/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"
MNIST_OMNI_LENET    = "/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet/LeNet_mnist_omni.pt"
PANICUM_RESNET18    = "/home/alexandreselani/Desktop/Experimento_panicum/ResNet18/Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt"
EUCALYPTUS_ALEXNET  = "/home/alexandreselani/Desktop/Eucalyptus/OpenSet/Models/{dataset}/AlexNet_fold_{fold}.pt"
TINYIMGNET_RESNET18 = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/Split_{split}/ResNet18_TinyImageNet_split_{split}.pt"

TINY_DATA_DIR   = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
TINY_SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"
TINY_NUM_KNOWN_CLASSES = 20
TINY_N_SPLITS = 5

EUCALYPTUS_TRANSFORMS = AlexNet_Weights.IMAGENET1K_V1.transforms()


# ── Nucleo do metodo ──────────────────────────────────────────────────────────
def coleta_softmax(model, loader):
    """Uma unica passada pela rede por (experimento, fold).

    Guarda, para cada amostra, a maior probabilidade da softmax, a classe predita e o
    rotulo verdadeiro. A varredura de thresholds depois so re-rotula essas saidas, sem
    repetir o forward 101 vezes.
    """
    model.eval()
    msp, preds, labels = [], [], []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            if isinstance(logits, (tuple, list)):   # backbones que devolvem (logits, feats)
                logits = logits[0]

            probs = torch.softmax(logits, dim=1)
            max_prob, pred = torch.max(probs, dim=1)

            msp.append(max_prob.detach().cpu())
            preds.append(pred.detach().cpu())
            labels.append(y.detach().cpu())

    msp = torch.cat(msp, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return msp, preds, labels


def avalia_threshold(msp, preds, labels, threshold):
    """Aplica o threshold na softmax e calcula as metricas com metricasImplementadasV2."""
    predicts = np.where(msp >= threshold, preds, -1)

    # Score de outlier: quanto maior, mais "desconhecida". A AUROC do
    # metricasImplementadasV2 usa 1 - outlier_scores, ou seja, a propria MSP.
    outlier_scores = 1.0 - msp

    metricas = metricasImplementadasV2(predict=predicts, label=labels,
                                       outlier_scores=outlier_scores, metodo="softmax")
    return metricas._metricas(), predicts


# ── Definicao dos experimentos ────────────────────────────────────────────────
class Experimento:
    """Amarra um backbone em disco aos loaders de validacao e teste do seu benchmark.

    n_folds = 0 indica benchmark de execucao unica (MNIST/Omniglot); nos demais, o
    threshold e escolhido pela media do F1 macro entre os folds/splits.
    """

    def __init__(self, nome, n_folds, build_model, build_val_loader, build_test_loader,
                 mc_column_names):
        self.nome = nome
        self.n_folds = n_folds
        self.build_model = build_model
        self.build_val_loader = build_val_loader
        self.build_test_loader = build_test_loader
        self.mc_column_names = mc_column_names

    @property
    def folds(self):
        return list(range(self.n_folds)) if self.n_folds > 0 else [0]


def carrega_pesos(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device=device)
    model.eval()
    return model


def _cache(fabrica):
    """Memoiza data managers: TinyImageNet, por exemplo, calcula mean/std de todos os
    splits na construcao, e isso nao precisa ser refeito entre validacao e teste."""
    guardado = {}

    def wrapper(*args):
        if args not in guardado:
            guardado[args] = fabrica(*args)
        return guardado[args]

    return wrapper


@_cache
def mnist_omni_manager(backbone):
    if backbone == "resnet18":
        transform = NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value
        bs = 256
    else:
        transform = NOMES.LENET_MNIST_OMNI_TRANSFORMS.value
        bs = 256
    return Mnist_omni_loader(bs=bs, transform=transform)


@_cache
def panicum_manager():
    return Panicum_halfsize_loader(bs=32)


@_cache
def eucalyptus_manager(dataset):
    return Eucalyptus_openset_loader(bs=32, dataset=dataset)


@_cache
def tinyimgnet_manager():
    return TinyImageNet_loader(data_dir=TINY_DATA_DIR, splits_dir=TINY_SPLITS_DIR,
                               batch_size=32, image_size=64)


def experimento_mnist_omni(backbone):
    """MNIST (conhecidas) x Omniglot (desconhecidas), execucao unica."""
    if backbone == "resnet18":
        pesos = MNIST_OMNI_RESNET18
        build_model = lambda fold: carrega_pesos(ResNet18(num_classes=10), pesos)
    else:
        pesos = MNIST_OMNI_LENET
        build_model = lambda fold: carrega_pesos(LeNet(num_classes=10), pesos)

    # load_gridsearch = 6000 MNIST val + 6000 Omniglot val; load_test = 10k MNIST + 10k Omniglot.
    return Experimento(
        nome=f"mnist_omni_{backbone}",
        n_folds=0,
        build_model=build_model,
        build_val_loader=lambda fold: mnist_omni_manager(backbone).load_gridsearch(),
        build_test_loader=lambda fold: mnist_omni_manager(backbone).load_test(),
        mc_column_names=["Omniglot"] + [str(i) for i in range(10)],
    )


def experimento_panicum():
    """Panicum (halfsize): Solo/Milho conhecidas, Panicum desconhecida, 5 folds."""
    transform = NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value

    return Experimento(
        nome="panicum",
        n_folds=5,
        build_model=lambda fold: carrega_pesos(
            ResNet18(num_classes=2), PANICUM_RESNET18.format(fold=fold)),
        build_val_loader=lambda fold: panicum_manager().load_val(fold, transform),
        build_test_loader=lambda fold: panicum_manager().load_test(fold, transform),
        mc_column_names=["Panicum", "Solo", "Milho"],
    )


def experimento_eucalyptus(dataset):
    """Eucalyptus open set: ground/healthy conhecidas, ceratocystis desconhecida, 5 folds."""
    return Experimento(
        nome=f"eucalyptus_{dataset}",
        n_folds=5,
        build_model=lambda fold: carrega_pesos(
            Alexnet(num_classes=2), EUCALYPTUS_ALEXNET.format(dataset=dataset, fold=fold)),
        build_val_loader=lambda fold: eucalyptus_manager(dataset).load_val(fold, EUCALYPTUS_TRANSFORMS),
        build_test_loader=lambda fold: eucalyptus_manager(dataset).load_test(fold, EUCALYPTUS_TRANSFORMS),
        mc_column_names=["Ceratocystis", "Ground", "Healthy"],
    )


def experimento_tinyimgnet():
    """Tiny ImageNet: 20 classes conhecidas por split, 5 splits."""
    def val_loader(split):
        manager = tinyimgnet_manager()
        return manager.get_val_loader(split, manager.eval_transforms[split])

    def test_loader(split):
        manager = tinyimgnet_manager()
        return manager.get_test_loader(split, manager.eval_transforms[split])

    return Experimento(
        nome="tinyimgnet",
        n_folds=TINY_N_SPLITS,
        build_model=lambda split: carrega_pesos(
            ResNet18_tinyimgnet(num_classes=TINY_NUM_KNOWN_CLASSES, weights=None),
            TINYIMGNET_RESNET18.format(split=split)),
        build_val_loader=val_loader,
        build_test_loader=test_loader,
        mc_column_names=["Unknown"] + [str(i) for i in range(TINY_NUM_KNOWN_CLASSES)],
    )


EXPERIMENTOS = {
    "mnist_omni_resnet18": lambda: experimento_mnist_omni("resnet18"),
    "panicum": experimento_panicum,
    "eucalyptus_dataset-1": lambda: experimento_eucalyptus("dataset-1"),
    "tinyimgnet": experimento_tinyimgnet,
}


# ── Etapas do protocolo ───────────────────────────────────────────────────────
def busca_threshold(exp):
    """Varre os thresholds na validacao e devolve o de maior F1 macro medio."""
    val_dir = os.path.join(OUTPUT_ROOT, exp.nome, "Val")

    # Sem matriz de confusao na varredura: seriam 101 matrizes por experimento, e so a
    # do threshold escolhido interessa (essa e gerada na etapa de teste).
    registra_metricas = metricLogger(THRESHOLDS, exp.n_folds, val_dir, flag_mc=False,
                                     mc_column_names=exp.mc_column_names, mc_title=exp.nome,
                                     predict_unknown_value=-1)

    f1_por_threshold = {t: [] for t in THRESHOLDS}

    for fold in exp.folds:
        gc.collect()
        torch.cuda.empty_cache()

        model = exp.build_model(fold)
        loader = exp.build_val_loader(fold)
        msp, preds, labels = coleta_softmax(model, loader)
        print(f"[{exp.nome}] fold/split {fold}: {len(labels)} amostras de validacao "
              f"({int((labels == -1).sum())} desconhecidas)")

        for threshold in THRESHOLDS:
            metricas, _ = avalia_threshold(msp, preds, labels, threshold)
            registra_metricas.update(metricas, fold, threshold)
            f1_por_threshold[threshold].append(metricas["F1 macro"])

        del model, loader
        gc.collect()
        torch.cuda.empty_cache()

    registra_metricas.aggregate("val_thresholds.csv")

    f1_medio = {t: float(np.mean(v)) for t, v in f1_por_threshold.items()}
    melhor_threshold = max(f1_medio, key=f1_medio.get)

    print(f"\n[{exp.nome}] melhor threshold na validacao: {melhor_threshold} "
          f"(F1 macro medio {f1_medio[melhor_threshold]:.4f})")
    return melhor_threshold, f1_medio[melhor_threshold]


def avalia_no_teste(exp, melhor_threshold):
    """Aplica o threshold escolhido no conjunto de teste."""
    test_dir = os.path.join(OUTPUT_ROOT, exp.nome, "Test")

    registra_metricas = metricLogger([melhor_threshold], exp.n_folds, test_dir, flag_mc=True,
                                     mc_column_names=exp.mc_column_names, mc_title=exp.nome,
                                     predict_unknown_value=-1)

    metricas_por_fold = []

    for fold in exp.folds:
        gc.collect()
        torch.cuda.empty_cache()

        model = exp.build_model(fold)
        loader = exp.build_test_loader(fold)
        msp, preds, labels = coleta_softmax(model, loader)

        metricas, predicts = avalia_threshold(msp, preds, labels, melhor_threshold)
        registra_metricas.update(metricas, fold, melhor_threshold)
        registra_metricas.update_mc(melhor_threshold, predicts, labels, labels)
        metricas_por_fold.append(metricas)

        print(f"[{exp.nome}] teste fold/split {fold}: F1 macro {metricas['F1 macro']:.4f}, "
              f"acc {metricas['accuracy']:.4f}, auroc {metricas['auroc']:.4f}")

        del model, loader
        gc.collect()
        torch.cuda.empty_cache()

    registra_metricas.aggregate("test_melhor_threshold.csv")
    return metricas_por_fold


def roda_experimento(nome):
    print("\n" + "=" * 70)
    print(f"EXPERIMENTO: {nome}  (baseline: threshold na softmax)")
    print("=" * 70)

    exp = EXPERIMENTOS[nome]()

    melhor_threshold, _ = busca_threshold(exp)
    metricas_por_fold = avalia_no_teste(exp, melhor_threshold)

    # O resumo guarda so o desempenho no teste; a varredura de validacao inteira ja esta
    # em Val/val_thresholds.csv.
    resumo = {"experiment": nome,
              "best_threshold": melhor_threshold}

    for chave in METRIC_KEYS:
        valores = [m[chave] for m in metricas_por_fold]
        resumo[f"{chave}_test_mean"] = float(np.mean(valores))
        resumo[f"{chave}_test_std"] = float(np.std(valores))

    return resumo


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experimentos", nargs="+", default=list(EXPERIMENTOS.keys()),
                        choices=list(EXPERIMENTOS.keys()),
                        help="Experimentos a executar (padrao: todos).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    resumos = [roda_experimento(nome) for nome in args.experimentos]

    df = pd.DataFrame(resumos)
    caminho = os.path.join(OUTPUT_ROOT, "resumo_softmax_threshold.csv")

    if os.path.exists(caminho):
        # Preserva no resumo os experimentos rodados antes que nao entraram nesta execucao.
        anterior = pd.read_csv(caminho)
        anterior = anterior[~anterior["experiment"].isin(df["experiment"])]
        pd.concat([anterior, df], ignore_index=True).to_csv(caminho, index=False, float_format="%.3f")
    else:
        df.to_csv(caminho, index=False, float_format="%.3f")

    print("\n" + "=" * 70)
    print("RESUMO (melhor threshold por experimento, metricas no teste)")
    print("=" * 70)
    print(df[["experiment", "best_threshold", "F1 macro_test_mean",
              "accuracy_test_mean", "auroc_test_mean"]].to_string(index=False))
    print(f"\nArquivo salvo: {caminho}")


if __name__ == "__main__":
    main()
