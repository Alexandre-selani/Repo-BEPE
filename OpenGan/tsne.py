import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
TINY_FEATS_DIR = Path("/home/alexandreselani/Desktop/Features_extraidas/Panicum_plantnet/ResNet18/Fold_1/")

N_SAMPLES_PER_SPLIT = 40  # samples to draw to keep t-SNE fast


def load_features(feats_dir: Path, suffix: str):
    """Loads features and labels saved by save_features()."""
    data = torch.load(feats_dir / f"{suffix}_features.pt", map_location="cpu")
    feats = data["features"]
    labels = data["labels"]
    return feats.numpy(), labels.numpy()


def subsample(feats, labels, n):
    """Random subsample to keep t-SNE tractable."""
    idx = np.random.choice(len(feats), size=min(n, len(feats)), replace=False)
    return feats[idx], labels[idx]


def main():
    np.random.seed(42)

    # ── Load TinyImageNet validation features ────────────────────────────────
    tiny_feats, tiny_labels = load_features(TINY_FEATS_DIR, "kkc_test")
    uuc_feats,uuc_labels = load_features(TINY_FEATS_DIR, "uuc_test")

    feats = np.concatenate([tiny_feats,uuc_feats])
    labels = np.concatenate([tiny_labels,uuc_labels])
    #tiny_feats, tiny_labels = subsample(tiny_feats, tiny_labels, N_SAMPLES_PER_SPLIT)

    print("Running t-SNE… (this may take a minute)")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=42,
        verbose=1
    )

    embedding = tsne.fit_transform(feats)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab20",
        s=10,
        alpha=0.7
    )

    ax.set_title("t-SNE: ResNet18 Features — TinyImageNet", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()

    out_path = Path("~/Desktop/paniucm_tsne.png").expanduser()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()