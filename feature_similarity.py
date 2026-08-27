"""
Similaridade visual das imagens de cada benchmark de Open Set Recognition, medida no
espaco de features da penultima camada do backbone ja treinado de cada experimento
(o mesmo modelo fechado que a baseline de softmax carrega do disco).

A ideia e caracterizar a geometria do espaco em que os metodos de OSR realmente
operam. Para cada fold/split reportamos tres medias de similaridade de cosseno:

  intra_kkc     pares de imagens da MESMA classe conhecida
  inter_kkc     pares de imagens de classes conhecidas DIFERENTES
  kkc_vs_uuc    pares entre o conjunto conhecido (tratado como uma unica classe) e o
                conjunto desconhecido (tratado como outra unica classe)

e no fim, por dataset, a media e o desvio padrao dessas medias entre os folds/splits.
Alem disso, a matriz completa de similaridade media entre todas as classes (as
conhecidas mais a desconhecida) e salva por dataset, que e o que mostra de qual classe
conhecida as desconhecidas se aproximam.

Como o backbone e o de cada experimento, as transforms de eval sao usadas exatamente
como no treino/avaliacao, com a normalizacao de cada benchmark.

Datasets:
  mnist_omni     MNIST x Omniglot, ResNet18 (10 classes)             - execucao unica
  panicum        Solo/Milho x Panicum, ResNet18 (2 classes)          - 5 folds
  tinyimgnet     20 conhecidas x 180 desconhecidas, ResNet18         - 5 splits
  ceratocystis   Ground/Healthy x Ceratocystis, AlexNet (2 classes)  - 5 folds
  (extras: mnist_omni_lenet, ceratocystis_dataset-20)

`--extrator dinov2` troca os backbones por um DINOv2 ViT-S/14 congelado, para comparar
o espaco aprendido no benchmark com o de um extrator generico. Nesse modo a
normalizacao de cada benchmark e substituida pela do ImageNet e as imagens sao
redimensionadas para 224x224 (ver `transform_para_dino`).

Uso:
    python feature_similarity.py                                  # backbones, teste
    python feature_similarity.py --datasets panicum tinyimgnet
    python feature_similarity.py --extrator dinov2 --conjunto val
"""

import argparse
import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import AlexNet_Weights

from Utils import fix_random_seed, NOMES
from Datasets import (Mnist_omni_loader, Panicum_halfsize_loader,
                      Eucalyptus_openset_loader, TinyImageNet_loader)
from Modelos import (ResNet18Featurizer, LeNetFeaturizer, AlexNetFeaturizer,
                     ResNet18_tinyimgnet_featurizer)

seed = 42
fix_random_seed(seed)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
UNKNOWN_LABEL = -1

OUTPUT_ROOT = "/home/alexandreselani/Desktop/Resultados_feature_similarity"

# ── Backbones ja treinados (mesmos pesos da baseline de softmax) ──────────────
MNIST_OMNI_RESNET18 = "/home/alexandreselani/Desktop/Experimento_mnist_omni/ResNet18/ResNet18_mnist_omni.pt"
MNIST_OMNI_LENET    = "/home/alexandreselani/Desktop/Experimento_mnist_omni/LeNet/LeNet_mnist_omni.pt"
PANICUM_RESNET18    = "/home/alexandreselani/Desktop/Experimento_panicum/ResNet18/Fold_{fold}/ResNet18_Panicum_fold_{fold}_plantnet.pt"
EUCALYPTUS_ALEXNET  = "/home/alexandreselani/Desktop/Eucalyptus/OpenSet/Models/{dataset}/AlexNet_fold_{fold}.pt"
TINYIMGNET_RESNET18 = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/ResNet18/Split_{split}/ResNet18_TinyImageNet_split_{split}.pt"

TINY_DATA_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/tiny-imagenet-200"
TINY_SPLITS_DIR = "/home/alexandreselani/Desktop/Experimento_tinyimgnet/data/class_splits"
TINY_N_SPLITS = 5
TINY_NUM_KNOWN_CLASSES = 20
TINY_IMAGE_SIZE = 64

EUCALYPTUS_TRANSFORMS = AlexNet_Weights.IMAGENET1K_V1.transforms()

# DINOv2 ViT-S/14 (modo de comparacao): entrada multipla de 14, normalizada com as
# estatisticas do ImageNet, com as quais o modelo foi pre-treinado.
DINO_REPO = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vits14"
DINO_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

METRICAS = ("intra_kkc", "inter_kkc", "kkc_vs_uuc")
METRICAS_LABEL = {
    "intra_kkc": "Intra-class (known)",
    "inter_kkc": "Inter-class (known)",
    "kkc_vs_uuc": "Known vs. unknown",
}
TITULO_EXTRATOR = {
    "backbone": "Penultimate-layer feature similarity (per-benchmark trained backbones)",
    "dinov2": "DINOv2 ViT-S/14 visual feature similarity",
}


# ── Transforms ────────────────────────────────────────────────────────────────
def _achata(transform):
    """Lista as operacoes de um Compose, expandindo Composes aninhados."""
    if isinstance(transform, transforms.Compose):
        ops = []
        for t in transform.transforms:
            ops.extend(_achata(t))
        return ops
    return [transform]


def transform_para_dino(transform):
    """Adapta a transform de eval de um benchmark para alimentar o DINOv2.

    So faz sentido no modo `--extrator dinov2`: com o backbone do proprio experimento a
    transform de eval e usada como esta.

    A geometria e a aparencia sao preservadas (mesmo resize, mesmo crop, mesmo
    grayscale, e a inversao de cores que o Mnist_omni_loader acrescenta depois desta
    transform), mas o Normalize final e removido, por dois motivos: cada benchmark
    normaliza com as suas proprias estatisticas (o Tiny ImageNet usa a media/desvio das
    classes conhecidas do split) e o DINOv2 espera as do ImageNet; e o RandomInvert do
    Omniglot so inverte corretamente um tensor em [0, 1]. O tensor sai em [0, 1], e o
    resize para 224x224 mais a normalizacao do ImageNet acontecem na GPU, em
    `preparo_dino`.

    Aceita um Compose ou os presets do torchvision (`AlexNet_Weights...transforms()`),
    que nao sao Composes.
    """
    if hasattr(transform, "crop_size") and hasattr(transform, "resize_size"):
        # Preset ImageClassification: resize do lado menor -> center crop -> tensor -> normalize.
        return transforms.Compose([
            transforms.Resize(transform.resize_size[0], interpolation=transform.interpolation),
            transforms.CenterCrop(transform.crop_size[0]),
            transforms.ToTensor(),
        ])

    return transforms.Compose(
        [t for t in _achata(transform) if not isinstance(t, transforms.Normalize)])


def prepara_transform(transform, extrator):
    return transform if extrator == "backbone" else transform_para_dino(transform)


# ── Modelos ───────────────────────────────────────────────────────────────────
def carrega_pesos(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device=DEVICE)
    model.eval()
    return model


_dino = {}


def carrega_dinov2():
    if "model" not in _dino:
        print(f"Carregando DINOv2 {DINO_MODEL} em {DEVICE}...")
        model = torch.hub.load(DINO_REPO, DINO_MODEL)
        model.to(DEVICE)
        model.eval()
        _dino["model"] = model
    return _dino["model"]


def modelo_ou_dino(extrator, fabrica):
    """No modo dinov2 o backbone do experimento e ignorado e o DINOv2 e reaproveitado."""
    if extrator == "dinov2":
        return lambda fold: carrega_dinov2()
    return fabrica


# ── Extracao de features ──────────────────────────────────────────────────────
_MEAN = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)


def preparo_dino(batch):
    """Fecha o que falta na GPU: 224x224, 3 canais e normalizacao do ImageNet."""
    if batch.shape[-2:] != (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE):
        batch = F.interpolate(batch, size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
                              mode="bilinear", align_corners=False, antialias=True)
    if batch.shape[1] == 1:
        batch = batch.expand(-1, 3, -1, -1)

    return (batch - _MEAN) / _STD


@torch.no_grad()
def extrai_features(model, loader, extrator, espaco="features"):
    """Representacao de cada imagem, em um de dois espacos.

    espaco="features" (padrao): a penultima camada, normalizada em L2, de modo que o
    produto interno ja e o cosseno - os 512 valores do avgpool na ResNet18, os 4096 do
    penultimo bloco do classificador na AlexNet, os 84 do fc2 na LeNet, o mesmo vetor
    que OpenMax, CAC e companhia usam. Os wrappers *Featurizer devolvem (logits, feats)
    e compartilham os nomes de camada dos backbones originais, entao carregam o
    state_dict treinado sem conversao.

    espaco="logits": a saida da camada de classificacao, SEM normalizar. E o espaco em
    que o MSP opera, e ali a magnitude e informacao: um logit alto e o que distingue uma
    amostra dentro do cone de uma classe de outra que so aponta na mesma direcao.
    Normalizar destruiria justamente isso. Como as estatisticas de similaridade deste
    modulo assumem vetores unitarios, este espaco e para visualizacao (ver
    plot_features_latent.py), nao para `estatisticas_similaridade`.
    """
    if espaco == "logits" and extrator == "dinov2":
        raise ValueError("O DINOv2 nao tem cabeca de classificacao: nao ha logits.")

    features, labels = [], []

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)

        if extrator == "dinov2":
            saida = model.forward_features(preparo_dino(images))["x_norm_clstoken"]
        else:
            logits, feats = model(images)
            saida = logits if espaco == "logits" else feats

        if espaco != "logits":
            saida = F.normalize(saida, dim=1)

        features.append(saida.cpu())
        labels.append(torch.as_tensor(targets).cpu())

    return torch.cat(features), torch.cat(labels).numpy()


# ── Similaridades ─────────────────────────────────────────────────────────────
def _somas_por_classe(features, labels):
    """Vetor soma e tamanho de cada classe presente em labels."""
    somas, tamanhos = {}, {}
    for classe in np.unique(labels):
        f_classe = features[torch.as_tensor(labels == classe)]
        somas[classe] = f_classe.sum(dim=0)
        tamanhos[classe] = f_classe.shape[0]
    return somas, tamanhos


def _media_intra(soma, n):
    """Media das similaridades dos pares distintos dentro de uma classe."""
    return (float(soma @ soma) - n) / (n * (n - 1)) if n > 1 else float("nan")


def _media_entre(soma_a, soma_b, n_a, n_b):
    return float(soma_a @ soma_b) / (n_a * n_b) if n_a and n_b else float("nan")


def estatisticas_similaridade(features, labels, unknown_label=UNKNOWN_LABEL):
    """Medias exatas das similaridades de cosseno, sem materializar a matriz de pares.

    Com as features normalizadas em L2, a soma das similaridades de um conjunto A
    depende apenas do vetor soma s_A = sum_i f_i:

        soma dos pares distintos dentro de A  =  (||s_A||^2 - n_A) / 2
        soma dos pares entre A e B (A != B)   =  s_A . s_B

    o que da as mesmas medias que percorrer os pares um a um, em O(n*d) de tempo e
    memoria em vez dos O(n^2) pares (no MNIST/Omniglot seriam 4x10^8 pares por fold).

    As medias "pooled" pesam cada par igualmente (equivalente a concatenar todos os
    pares e tirar a media); as versoes "_macro" tiram a media por classe (ou por par
    de classes) antes de agregar, ignorando o desbalanceamento.
    """
    features = features.double()
    labels = np.asarray(labels)

    mask_kkc = torch.as_tensor(labels != unknown_label)
    classes_kkc = np.unique(labels[labels != unknown_label])
    somas, tamanhos = _somas_por_classe(features, labels)

    # Intra-classe: apenas classes conhecidas, pares distintos (sem a auto-similaridade).
    soma_intra = pares_intra = 0.0
    macro_intra = []
    for classe in classes_kkc:
        n = tamanhos[classe]
        if n < 2:
            continue
        soma_intra += float(somas[classe] @ somas[classe]) - n
        pares_intra += n * (n - 1)
        macro_intra.append(_media_intra(somas[classe], n))

    # Inter-classe: cada par de classes conhecidas distintas, contado uma unica vez.
    soma_inter = pares_inter = 0.0
    macro_inter = []
    for i, classe_a in enumerate(classes_kkc):
        for classe_b in classes_kkc[i + 1:]:
            soma_inter += float(somas[classe_a] @ somas[classe_b])
            pares_inter += tamanhos[classe_a] * tamanhos[classe_b]
            macro_inter.append(_media_entre(somas[classe_a], somas[classe_b],
                                            tamanhos[classe_a], tamanhos[classe_b]))

    # Conhecidas (uma unica classe) x desconhecidas (outra unica classe).
    n_kkc = int(mask_kkc.sum())
    n_uuc = int((~mask_kkc).sum())
    kkc_vs_uuc = _media_entre(features[mask_kkc].sum(dim=0), features[~mask_kkc].sum(dim=0),
                              n_kkc, n_uuc)

    return {
        "intra_kkc": soma_intra / pares_intra if pares_intra else float("nan"),
        "inter_kkc": soma_inter / pares_inter if pares_inter else float("nan"),
        "kkc_vs_uuc": kkc_vs_uuc,
        "intra_kkc_macro": float(np.mean(macro_intra)) if macro_intra else float("nan"),
        "inter_kkc_macro": float(np.mean(macro_inter)) if macro_inter else float("nan"),
        "n_imagens": n_kkc + n_uuc,
        "n_kkc": n_kkc,
        "n_uuc": n_uuc,
        "n_classes_kkc": len(classes_kkc),
    }


def matriz_por_classe(features, labels):
    """Similaridade media de cada par de classes, com a desconhecida como mais uma.

    A diagonal e a similaridade intra-classe. E dela que se le de qual classe conhecida
    as desconhecidas estao proximas, coisa que o kkc_vs_uuc agregado esconde.
    """
    features = features.double()
    labels = np.asarray(labels)
    somas, tamanhos = _somas_por_classe(features, labels)

    valores = {}
    for i, a in enumerate(sorted(somas)):
        for b in sorted(somas)[i:]:
            valores[(a, b)] = (_media_intra(somas[a], tamanhos[a]) if a == b
                               else _media_entre(somas[a], somas[b], tamanhos[a], tamanhos[b]))
    return valores, tamanhos


# ── Definicao dos datasets ────────────────────────────────────────────────────
class Experimento:
    """Amarra um benchmark ao backbone e ao loader de cada um dos seus folds/splits.

    n_folds = 0 indica benchmark de execucao unica (MNIST/Omniglot), tratado como um
    unico fold: a media entre folds e o proprio valor e o desvio padrao e zero.
    """

    def __init__(self, nome, n_folds, build_model, build_loader, descricao, rotulos=None):
        self.nome = nome
        self.n_folds = n_folds
        self.build_model = build_model
        self.build_loader = build_loader
        self.descricao = descricao
        self.rotulos = rotulos or {}

    @property
    def folds(self):
        return list(range(self.n_folds)) if self.n_folds > 0 else [0]

    def rotulo(self, classe):
        return self.rotulos.get(classe, "UUC" if classe == UNKNOWN_LABEL else str(classe))


def _cache(fabrica):
    """Memoiza data managers: o TinyImageNet_loader calcula a media/desvio de todos os
    splits na construcao, e isso nao precisa ser refeito a cada fold."""
    guardado = {}

    def wrapper(*args):
        if args not in guardado:
            guardado[args] = fabrica(*args)
        return guardado[args]

    return wrapper


@_cache
def mnist_omni_manager(backbone, extrator):
    transform = (NOMES.RESNET18_MNIST_OMNI_EVAL_TRANSFORMS.value if backbone == "resnet18"
                 else NOMES.LENET_MNIST_OMNI_TRANSFORMS.value)

    # invert_omni_colors=True: sem isso a intensidade media do pixel ja separa
    # conhecido de desconhecido, e a similaridade mediria esse atalho, nao a forma.
    return Mnist_omni_loader(bs=BATCH_SIZE, transform=prepara_transform(transform, extrator),
                             invert_omni_colors=True)


@_cache
def panicum_manager():
    return Panicum_halfsize_loader(bs=BATCH_SIZE)


@_cache
def eucalyptus_manager(dataset):
    return Eucalyptus_openset_loader(bs=BATCH_SIZE, dataset=dataset)


@_cache
def tinyimgnet_manager():
    return TinyImageNet_loader(data_dir=TINY_DATA_DIR, splits_dir=TINY_SPLITS_DIR,
                               batch_size=BATCH_SIZE, image_size=TINY_IMAGE_SIZE)


def experimento_mnist_omni(conjunto, extrator, backbone="resnet18"):
    """MNIST (conhecidas) x Omniglot (desconhecida), execucao unica.

    load_test = 10k MNIST + 10k Omniglot; load_gridsearch = 6k MNIST val + 6k Omniglot val.
    """
    if backbone == "resnet18":
        fabrica = lambda fold: carrega_pesos(ResNet18Featurizer(num_classes=10),
                                             MNIST_OMNI_RESNET18)
    else:
        fabrica = lambda fold: carrega_pesos(LeNetFeaturizer(num_classes=10),
                                             MNIST_OMNI_LENET)

    def loader(fold):
        manager = mnist_omni_manager(backbone, extrator)
        return manager.load_test() if conjunto == "test" else manager.load_gridsearch()

    nome = "mnist_omni" if backbone == "resnet18" else f"mnist_omni_{backbone}"
    return Experimento(
        nome=nome, n_folds=0,
        build_model=modelo_ou_dino(extrator, fabrica), build_loader=loader,
        descricao=f"MNIST x Omniglot ({backbone})",
        rotulos={-1: "omniglot(UUC)"})


def experimento_panicum(conjunto, extrator):
    """Panicum (halfsize): Solo/Milho conhecidas, Panicum desconhecida, 5 folds."""
    transform = prepara_transform(NOMES.PANICUM_PLANTNET_VAL_TRANSFORMS.value, extrator)
    fabrica = lambda fold: carrega_pesos(ResNet18Featurizer(num_classes=2),
                                         PANICUM_RESNET18.format(fold=fold))

    def loader(fold):
        manager = panicum_manager()
        return (manager.load_test(fold, transform) if conjunto == "test"
                else manager.load_val(fold, transform))

    return Experimento(
        nome="panicum", n_folds=5,
        build_model=modelo_ou_dino(extrator, fabrica), build_loader=loader,
        descricao="Solo/Milho x Panicum (ResNet18)",
        rotulos={0: "solo", 1: "milho", -1: "panicum(UUC)"})


def experimento_ceratocystis(conjunto, extrator, dataset="dataset-1"):
    """Eucalyptus open set: ground/healthy conhecidas, ceratocystis desconhecida, 5 folds."""
    transform = prepara_transform(EUCALYPTUS_TRANSFORMS, extrator)
    fabrica = lambda fold: carrega_pesos(
        AlexNetFeaturizer(num_classes=2), EUCALYPTUS_ALEXNET.format(dataset=dataset, fold=fold))

    def loader(fold):
        manager = eucalyptus_manager(dataset)
        return (manager.load_test(fold, transform) if conjunto == "test"
                else manager.load_val(fold, transform))

    nome = "ceratocystis" if dataset == "dataset-1" else f"ceratocystis_{dataset}"
    return Experimento(
        nome=nome, n_folds=5,
        build_model=modelo_ou_dino(extrator, fabrica), build_loader=loader,
        descricao=f"Ground/Healthy x Ceratocystis, {dataset} (AlexNet)",
        rotulos={0: "ground", 1: "healthy", -1: "ceratocystis(UUC)"})


def experimento_tinyimgnet(conjunto, extrator):
    """Tiny ImageNet: 20 classes conhecidas e 180 desconhecidas por split, 5 splits."""
    fabrica = lambda split: carrega_pesos(
        ResNet18_tinyimgnet_featurizer(num_classes=TINY_NUM_KNOWN_CLASSES, weights=None),
        TINYIMGNET_RESNET18.format(split=split))

    def loader(split):
        manager = tinyimgnet_manager()
        transform = prepara_transform(manager.eval_transforms[split], extrator)
        return (manager.get_test_loader(split, transform) if conjunto == "test"
                else manager.get_val_loader(split, transform))

    return Experimento(
        nome="tinyimgnet", n_folds=TINY_N_SPLITS,
        build_model=modelo_ou_dino(extrator, fabrica), build_loader=loader,
        descricao="20 conhecidas x 180 desconhecidas (ResNet18)")


EXPERIMENTOS = {
    "mnist_omni": experimento_mnist_omni,
    "panicum": experimento_panicum,
    "tinyimgnet": experimento_tinyimgnet,
    "ceratocystis": experimento_ceratocystis,
    "mnist_omni_lenet": lambda c, e: experimento_mnist_omni(c, e, "lenet"),
    "ceratocystis_dataset-20": lambda c, e: experimento_ceratocystis(c, e, "dataset-20"),
}


# ── Execucao ──────────────────────────────────────────────────────────────────
def roda_experimento(exp, extrator):
    print("\n" + "=" * 78)
    print(f"DATASET: {exp.nome}  ({exp.descricao})")
    print("=" * 78)

    linhas, matrizes, tamanhos_por_classe = [], {}, {}

    for fold in exp.folds:
        gc.collect()
        torch.cuda.empty_cache()

        model = exp.build_model(fold)
        loader = exp.build_loader(fold)
        features, labels = extrai_features(model, loader, extrator)

        estatisticas = estatisticas_similaridade(features, labels)
        linhas.append({"dataset": exp.nome, "fold": fold, **estatisticas})

        valores, tamanhos = matriz_por_classe(features, labels)
        for par, valor in valores.items():
            matrizes.setdefault(par, []).append(valor)
        for classe, n in tamanhos.items():
            tamanhos_por_classe.setdefault(classe, []).append(n)

        print(f"[{exp.nome}] fold/split {fold}: {estatisticas['n_imagens']} imagens "
              f"({estatisticas['n_kkc']} conhecidas em {estatisticas['n_classes_kkc']} classes, "
              f"{estatisticas['n_uuc']} desconhecidas) | "
              f"intra {estatisticas['intra_kkc']:.4f} | "
              f"inter {estatisticas['inter_kkc']:.4f} | "
              f"kkc x uuc {estatisticas['kkc_vs_uuc']:.4f}")

        del model, loader, features, labels
        gc.collect()
        torch.cuda.empty_cache()

    resumo = {"dataset": exp.nome, "n_folds": len(exp.folds)}
    for chave in METRICAS + ("intra_kkc_macro", "inter_kkc_macro"):
        valores = [linha[chave] for linha in linhas]
        resumo[f"{chave}_mean"] = float(np.mean(valores))
        resumo[f"{chave}_std"] = float(np.std(valores))

    return linhas, resumo, matriz_como_dataframe(exp, matrizes, tamanhos_por_classe)


def matriz_como_dataframe(exp, matrizes, tamanhos_por_classe):
    """Matriz classe x classe (media entre folds) em formato longo, com o desvio."""
    linhas = []
    for (a, b), valores in matrizes.items():
        linhas.append({
            "dataset": exp.nome,
            "classe_a": exp.rotulo(a),
            "classe_b": exp.rotulo(b),
            "tipo": "intra" if a == b else ("kkc_x_uuc" if UNKNOWN_LABEL in (a, b) else "kkc_x_kkc"),
            "sim_mean": float(np.mean(valores)),
            "sim_std": float(np.std(valores)),
            "n_a": float(np.mean(tamanhos_por_classe[a])),
            "n_b": float(np.mean(tamanhos_por_classe[b])),
        })
    return pd.DataFrame(linhas)


def imprime_matriz(exp, df):
    """Mostra a matriz classe x classe do dataset no terminal."""
    classes = sorted(set(df["classe_a"]) | set(df["classe_b"]))
    if len(classes) > 6:      # tinyimgnet: 21 classes nao cabem no terminal, so no CSV
        return

    valores = {}
    for _, linha in df.iterrows():
        valores[(linha["classe_a"], linha["classe_b"])] = linha["sim_mean"]
        valores[(linha["classe_b"], linha["classe_a"])] = linha["sim_mean"]

    print(f"\n  matriz classe x classe ({exp.nome}, media entre {len(exp.folds)} folds)")
    print("  " + " " * 20 + "".join(f"{c:>20}" for c in classes))
    for a in classes:
        print("  " + f"{a:<20}" + "".join(f"{valores[(a, b)]:>20.4f}" for b in classes))


def grafico_resumo(df, caminho, extrator):
    """Barras agrupadas: as tres medias por dataset, com o desvio padrao entre folds."""
    datasets = df["dataset"].tolist()
    x = np.arange(len(datasets))
    largura = 0.26

    plt.figure(figsize=(1.9 * len(datasets) + 4, 5))

    for i, metrica in enumerate(METRICAS):
        plt.bar(x + (i - 1) * largura, df[f"{metrica}_mean"], largura,
                yerr=df[f"{metrica}_std"], capsize=4, label=METRICAS_LABEL[metrica])

    plt.xticks(x, datasets)
    plt.ylabel("Cosine similarity")
    plt.title(f"{TITULO_EXTRATOR[extrator]}\n(mean $\\pm$ std across folds/splits)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()

    print(f"Grafico salvo: {caminho}")


def salva(df, caminho, chave="dataset"):
    """Preserva no arquivo as linhas de datasets que nao entraram nesta execucao."""
    if os.path.exists(caminho):
        anterior = pd.read_csv(caminho)
        anterior = anterior[~anterior[chave].isin(df[chave])]
        df = pd.concat([anterior, df], ignore_index=True)

    df.to_csv(caminho, index=False, float_format="%.4f")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        default=["mnist_omni", "panicum", "tinyimgnet", "ceratocystis"],
                        choices=list(EXPERIMENTOS.keys()),
                        help="Datasets a analisar (padrao: os quatro benchmarks).")
    parser.add_argument("--conjunto", choices=["test", "val"], default="test",
                        help="Split avaliado em cada fold (padrao: test).")
    parser.add_argument("--extrator", choices=["backbone", "dinov2"], default="backbone",
                        help="Espaco de features: o backbone treinado de cada experimento "
                             "(padrao) ou um DINOv2 ViT-S/14 congelado.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    sufixo = f"{args.extrator}_{args.conjunto}"

    linhas, resumos, matrizes = [], [], []
    for nome in args.datasets:
        exp = EXPERIMENTOS[nome](args.conjunto, args.extrator)
        fold_linhas, resumo, matriz = roda_experimento(exp, args.extrator)
        imprime_matriz(exp, matriz)
        linhas += fold_linhas
        resumos.append(resumo)
        matrizes.append(matriz)

    salva(pd.DataFrame(linhas), os.path.join(OUTPUT_ROOT, f"similaridade_por_fold_{sufixo}.csv"))
    salva(pd.concat(matrizes, ignore_index=True),
          os.path.join(OUTPUT_ROOT, f"matriz_classes_{sufixo}.csv"))
    df_resumo = salva(pd.DataFrame(resumos),
                      os.path.join(OUTPUT_ROOT, f"resumo_similaridade_{sufixo}.csv"))

    grafico_resumo(df_resumo, os.path.join(OUTPUT_ROOT, f"similaridade_{sufixo}.png"),
                   args.extrator)

    print("\n" + "=" * 78)
    print(f"RESUMO - media +/- desvio padrao entre folds/splits "
          f"(extrator: {args.extrator}, conjunto: {args.conjunto})")
    print("=" * 78)
    print(f"{'dataset':<26}{'folds':>7}"
          f"{'intra (KKC)':>20}{'inter (KKC)':>20}{'KKC x UUC':>20}")
    print("-" * 78)
    for _, linha in df_resumo.iterrows():
        print(f"{linha['dataset']:<26}{int(linha['n_folds']):>7}"
              + "".join(f"{linha[f'{m}_mean']:>12.4f} +/-{linha[f'{m}_std']:>6.4f}"
                        for m in METRICAS))
    print("=" * 78)
    print(f"\nArquivos salvos em: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
