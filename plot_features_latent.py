"""
Figura do espaco latente dos quatro benchmarks (fig:plots_features).

Mostra o espaco de logits do backbone treinado de cada experimento, no primeiro
fold/split, para contrastar os benchmarks agricolas (poucas classes conhecidas, bem
separadas, desconhecida em regiao propria) com MNIST/Omniglot e TinyImageNet. E o
espaco em que o MSP decide: a softmax e uma funcao apenas dos logits.

Os logits nao sao normalizados. A magnitude e informacao aqui - e o que separa uma
amostra dentro do cone de uma classe de outra que so aponta na mesma direcao - e
normalizar apagaria exatamente o sinal que o MSP (e o MLS de Vaze et al.) usa.

Como o espaco de logits tem uma dimensao por classe conhecida, os benchmarks de duas
classes sao plotados exatamente, sem projecao: os eixos sao os proprios logits. Com
mais classes cai-se no t-SNE euclidiano (`--projecao tsne` forca o t-SNE em todos os
paineis, se a uniformidade for preferivel).

`--espaco features` volta a projetar a penultima camada normalizada em L2, a mesma
geometria que o feature_similarity.py resume em numeros.

Uso:
    python plot_features_latent.py
    python plot_features_latent.py --espaco features
    python plot_features_latent.py --conjunto val --max-por-classe 400
"""

import argparse
import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from feature_similarity import (EXPERIMENTOS, OUTPUT_ROOT, UNKNOWN_LABEL, extrai_features)

SEED = 42

# Titulo de cada painel; a letra ((a), (b), ...) e atribuida na ordem em que os
# datasets forem pedidos, para a numeracao continuar certa quando um painel sai.
PAINEIS = {
    "panicum": "Panicum - ResNet18",
    "ceratocystis": "Ceratocystis - AlexNet",
    "mnist_omni": "MNIST / Omniglot - ResNet18",
    "tinyimgnet": "TinyImageNet - ResNet18",
}

# Nomes de classe para a legenda; o que faltar cai no rotulo do proprio Experimento.
ROTULOS = {
    "panicum": {0: "Soil", 1: "Maize", UNKNOWN_LABEL: "Panicum (unknown)"},
    "ceratocystis": {0: "Ground", 1: "Healthy", UNKNOWN_LABEL: "Ceratocystis (unknown)"},
    "mnist_omni": {**{i: f"Digit {i}" for i in range(10)},
                   UNKNOWN_LABEL: "Omniglot (unknown)"},
    "tinyimgnet": {UNKNOWN_LABEL: "Unknown (180 classes)"},
}

COR_UUC = "black"
MAX_CLASSES_NA_LEGENDA = 11


def disposicao(n_paineis):
    """Uma unica linha ate tres paineis; a partir dai, duas colunas."""
    if n_paineis <= 3:
        return 1, n_paineis, (5.3 * n_paineis, 5.6)

    n_linhas = -(-n_paineis // 2)
    return n_linhas, 2, (11, 4.75 * n_linhas)


def amostra_estratificada(labels, max_por_classe, rng):
    """Limita os pontos do t-SNE mantendo a proporcao conhecido/desconhecido do split.

    O MNIST/Omniglot tem 20k pontos no teste, ilegiveis e lentos de projetar, e o
    desenho da estrutura nao muda com uma amostra. As conhecidas sao limitadas por
    classe; o desconhecido, que e uma classe so mas concentra metade das imagens, e
    reamostrado na mesma razao do split original - com um teto por classe ele apareceria
    dez vezes menor do que e no MNIST/Omniglot.
    """
    def sorteia(candidatos, quantos):
        return rng.choice(candidatos, quantos, replace=False) if len(candidatos) > quantos \
            else candidatos

    classes_kkc = [c for c in np.unique(labels) if c != UNKNOWN_LABEL]
    indices = [sorteia(np.flatnonzero(labels == c), max_por_classe) for c in classes_kkc]

    n_kkc = int((labels != UNKNOWN_LABEL).sum())
    n_uuc = int((labels == UNKNOWN_LABEL).sum())
    if n_uuc:
        amostrados_kkc = sum(len(i) for i in indices)
        alvo = int(round(amostrados_kkc * n_uuc / n_kkc)) if n_kkc else n_uuc
        indices.append(sorteia(np.flatnonzero(labels == UNKNOWN_LABEL), max(alvo, 1)))

    return np.sort(np.concatenate(indices))


def projeta(dados, espaco, projecao):
    """Devolve os pontos 2D do painel e o rotulo dos eixos.

    Com dois logits nao ha o que projetar: o espaco ja e o plano do grafico, e mostra-lo
    direto evita a distorcao que o t-SNE introduziria de graca. Nos demais casos, t-SNE
    - euclidiano nos logits, onde a magnitude conta, e cosseno nas features, que ja
    estao na esfera unitaria.
    """
    if espaco == "logits" and dados.shape[1] == 2 and projecao == "auto":
        return dados.numpy(), "logit"

    n = dados.shape[0]
    tsne = TSNE(n_components=2, metric="euclidean" if espaco == "logits" else "cosine",
                init="pca", perplexity=min(30.0, max(5.0, (n - 1) / 4)),
                max_iter=1000, random_state=SEED)
    return tsne.fit_transform(dados.numpy()), "t-SNE"


def desenha_painel(ax, pontos, labels, nome, titulo, eixos):
    rotulos = ROTULOS.get(nome, {})
    classes_kkc = sorted(c for c in np.unique(labels) if c != UNKNOWN_LABEL)

    # tab10 quando as classes cabem na legenda, tab20 no TinyImageNet.
    cmap = plt.get_cmap("tab10" if len(classes_kkc) <= 10 else "tab20")
    legenda_por_classe = len(classes_kkc) + 1 <= MAX_CLASSES_NA_LEGENDA

    for i, classe in enumerate(classes_kkc):
        mask = labels == classe
        ax.scatter(pontos[mask, 0], pontos[mask, 1], s=9, alpha=0.75, linewidths=0,
                   color=cmap(i % cmap.N),
                   label=rotulos.get(classe, str(classe)) if legenda_por_classe else None)

    if not legenda_por_classe:
        # Uma unica entrada cinza representando as 20 conhecidas, que estao coloridas.
        ax.scatter([], [], s=9, color="grey", label=f"Known ({len(classes_kkc)} classes)")

    mask = labels == UNKNOWN_LABEL
    ax.scatter(pontos[mask, 0], pontos[mask, 1], s=13, alpha=0.55, marker="x",
               linewidths=0.7, color=COR_UUC,
               label=rotulos.get(UNKNOWN_LABEL, "Unknown"))

    ax.set_title(titulo, fontsize=12)

    if eixos == "logit":
        # Eixos com escala: nos paineis de duas classes o valor do logit e legivel.
        ax.set_xlabel("Logit $z_0$", fontsize=11)
        ax.set_ylabel("Logit $z_1$", fontsize=11)
        ax.tick_params(labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.legend(loc="best", fontsize=11, markerscale=1.6, framealpha=0.85,
              ncol=2 if len(classes_kkc) > 4 else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conjunto", choices=["test", "val"], default="test")
    parser.add_argument("--datasets", nargs="+", default=list(PAINEIS),
                        choices=list(PAINEIS),
                        help="Paineis da figura, na ordem (padrao: os quatro).")
    parser.add_argument("--nome-saida", default=None,
                        help="Nome do arquivo sem extensao (padrao: derivado do espaco "
                             "e do conjunto). Use para nao sobrescrever outra versao.")
    parser.add_argument("--espaco", choices=["logits", "features"], default="logits",
                        help="Espaco mostrado: os logits, onde o MSP decide (padrao), "
                             "ou a penultima camada normalizada em L2.")
    parser.add_argument("--projecao", choices=["auto", "tsne"], default="auto",
                        help="auto (padrao): plota os logits sem projecao quando ha so "
                             "duas classes conhecidas; tsne: projeta todos os paineis.")
    parser.add_argument("--max-por-classe", type=int, default=300,
                        help="Pontos por classe no grafico (padrao: 300).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    n_linhas, n_colunas, figsize = disposicao(len(args.datasets))
    fig, axes = plt.subplots(n_linhas, n_colunas, figsize=figsize, squeeze=False)

    for i, (ax, nome) in enumerate(zip(axes.ravel(), args.datasets)):
        titulo = f"({chr(ord('a') + i)}) {PAINEIS[nome]}"

        exp = EXPERIMENTOS[nome](args.conjunto, "backbone")
        fold = exp.folds[0]

        model = exp.build_model(fold)
        loader = exp.build_loader(fold)
        dados, labels = extrai_features(model, loader, "backbone", args.espaco)

        indices = amostra_estratificada(labels, args.max_por_classe, rng)
        dados, labels = dados[indices], labels[indices]

        pontos, eixos = projeta(dados, args.espaco, args.projecao)

        print(f"[{nome}] fold/split {fold}: {len(labels)} pontos "
              f"({int((labels != UNKNOWN_LABEL).sum())} conhecidos, "
              f"{int((labels == UNKNOWN_LABEL).sum())} desconhecidos), "
              f"dim {dados.shape[1]} -> {eixos}")

        desenha_painel(ax, pontos, labels, nome, titulo, eixos)

        del model, loader, dados, labels
        gc.collect()
        torch.cuda.empty_cache()

    subtitulo = ("Logit space" if args.espaco == "logits"
                 else "Penultimate-layer features (t-SNE, cosine)")
    fig.suptitle(f"{subtitulo}, first fold/split of each benchmark", fontsize=12)
    fig.tight_layout()

    nome_saida = args.nome_saida or f"plots_features_{args.espaco}_{args.conjunto}"
    caminho = os.path.join(OUTPUT_ROOT, f"{nome_saida}.png")
    fig.savefig(caminho, dpi=300)
    fig.savefig(caminho.replace(".png", ".pdf"))
    plt.close(fig)

    print(f"\nFigura salva: {caminho} (e o .pdf ao lado)")


if __name__ == "__main__":
    main()
