import numpy as np
import torch
import pickle
from app.sdf import SparseDataFrame
from app.classification import _calc_embeddings
from app import logger
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


sns.set_style("ticks")


def load_model(path):
    model = torch.load(path, map_location="cpu")
    return model


def load_data(sdf_train_path):
    with open(sdf_train_path, "rb") as f:
        sdf_train = pickle.load(f)
    return sdf_train


"""
K01866               YARS, tyrS; tyrosyl-tRNA synthetase [EC:6.1.1.1]
K01867               WARS, trpS; tryptophanyl-tRNA synthetase [EC:6.1.1.2]
K01868               TARS, thrS; threonyl-tRNA synthetase [EC:6.1.1.3]
K01869               LARS, leuS; leucyl-tRNA synthetase [EC:6.1.1.4]
K01870               IARS, ileS; isoleucyl-tRNA synthetase [EC:6.1.1.5]
K01872               AARS, alaS; alanyl-tRNA synthetase [EC:6.1.1.7]
K01873               VARS, valS; valyl-tRNA synthetase [EC:6.1.1.9]
K01874               MARS, metG; methionyl-tRNA synthetase [EC:6.1.1.10]
K01875               SARS, serS; seryl-tRNA synthetase [EC:6.1.1.11]
K01876               DARS2, aspS; aspartyl-tRNA synthetase [EC:6.1.1.12]
"""
kos = [
    "K01866",
    "K01867",
    "K01868",
    "K01869",
    "K01870",
    "K01872",
    "K01873",
    "K01874",
    "K01875",
    "K01876",
]

labels = [
    "Tyrosyl-tRNA synthetase",
    "Tryptophanyl-tRNA synthetase",
    "Threonyl-tRNA synthetase",
    "Leucyl-tRNA synthetase",
    "Isoleucyl-tRNA synthetase",
    "Alanyl-tRNA synthetase",
    "Valyl-tRNA synthetase",
    "Methionyl-tRNA synthetase",
    "Seryl-tRNA synthetase",
    "Aspartyl-tRNA synthetase",
]

ko_to_label = dict(zip(kos, labels))

colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

print("Started")

sdf_train_path = "data/kegg_2021_coverage_0.8/sdf_train.pkl"
model_path = "tmp/model.pt"
logger.info("Loading model")
model = load_model(model_path)
sdf_train: SparseDataFrame = load_data(sdf_train_path)
sdf_train = sdf_train.select_by_labels(kos)


assert all(any(l in set(kos) for l in labels) for labels in sdf_train.labels.values())


sdf_train.labels = np.array([sdf_train.labels[k][0] for k in sdf_train.index_ids])
sdf_train.labels = np.array([ko_to_label[k] for k in sdf_train.labels])

# ============================== PCA ==============================

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

raw_data = sdf_train.matrix.toarray()
pca = PCA(n_components=2)
pca.fit(raw_data)
pca_data = pca.transform(raw_data)

sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=sdf_train.labels,
    palette=colors,
    ax=axes[0, 0],
    legend=False,
    alpha=0.5,
)
axes[0, 0].set_xlabel("PC1", fontsize=20)
axes[0, 0].set_ylabel("PC2", fontsize=20)
axes[0, 0].grid(False)
for spine in axes[0, 0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")

# annotate plot with 'A'
axes[0, 0].annotate(
    "A",
    xy=(-0.2, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)

# l2 normalize raw data
# norm_data = raw_data / np.linalg.norm(raw_data, axis=1)[:, None]
# pca = PCA(n_components=2)
# pca.fit(norm_data)
# pca_norm = pca.transform(norm_data)
# sns.scatterplot(
#     x=pca_norm[:, 0],
#     y=pca_norm[:, 1],
#     hue=sdf_train.labels,
#     palette=colors,
#     ax=axes[0, 1],
#     legend=False,
#     alpha=0.5,
# )

# axes[0, 1].set_xlabel("PC1", fontsize=20)
# axes[0, 1].set_ylabel("PC2", fontsize=20)
# axes[0, 1].grid(False)
# for spine in axes[0, 1].spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(1)
#     spine.set_color("black")
# axes[0, 1].annotate(
#     "B",
#     xy=(-0.2, 0.9),
#     xycoords="axes fraction",
#     fontsize=16,
#     horizontalalignment="left",
#     verticalalignment="bottom",
# )


logger.info("Calculating embeddings")
embeddings = _calc_embeddings(sdf=sdf_train, model=model, device="cpu")
embeddings = embeddings.numpy()
logger.info("Plotting")
pca = PCA(n_components=2)
pca.fit(embeddings)
pca_embeddings = pca.transform(embeddings)
sns.scatterplot(
    x=pca_embeddings[:, 0],
    y=pca_embeddings[:, 1],
    hue=sdf_train.labels,
    palette=colors,
    ax=axes[1, 0],
    alpha=0.5,
    legend=False,
)
axes[1, 0].set_xlabel("PC1", fontsize=20)
axes[1, 0].set_ylabel("PC2", fontsize=20)
axes[1, 0].grid(False)
for spine in axes[1, 0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
axes[1, 0].annotate(
    "C",
    xy=(-0.2, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)
artists = []
for i, label in enumerate(labels):
    artists.append(
        plt.Line2D(
            (0, 1),
            (0, 0),
            color=colors[i],
            label=label,
            marker="o",
            linestyle="",
            markersize=10,
        )
    )
plt.legend(
    handles=artists,
    bbox_to_anchor=(2.5, 1),
    loc="center right",
    borderaxespad=0.0,
    fontsize=16,
    frameon=False,
)
# increase spacing between subplots
plt.subplots_adjust(wspace=0.3)
plt.savefig("pca.png", dpi=300, bbox_inches="tight")

exit()
# ============================== t-SNE ==============================
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

raw_data = sdf_train.matrix.toarray()
tsne = TSNE(n_components=2)
tsne.fit(raw_data)
tsne_data = tsne.fit_transform(raw_data)

sns.scatterplot(
    x=tsne_data[:, 0],
    y=tsne_data[:, 1],
    hue=sdf_train.labels,
    palette=colors,
    ax=axes[0, 0],
    legend=False,
    alpha=0.5,
)
axes[0, 0].set_xlabel("t-SNE 1", fontsize=20)
axes[0, 0].set_ylabel("t-SNE 2", fontsize=20)
axes[0, 0].grid(False)
for spine in axes[0, 0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")

# annotate plot with 'A'
axes[0, 0].annotate(
    "A",
    xy=(-0.2, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)

# l2 normalize raw data
norm_data = raw_data / np.linalg.norm(raw_data, axis=1)[:, None]
tsne = TSNE(n_components=2)
tsne.fit(norm_data)
tsne_norm = tsne.fit_transform(norm_data)
sns.scatterplot(
    x=tsne_norm[:, 0],
    y=tsne_norm[:, 1],
    hue=sdf_train.labels,
    palette=colors,
    ax=axes[0, 1],
    legend=False,
    alpha=0.5,
)

axes[0, 1].set_xlabel("t-SNE 1", fontsize=20)
axes[0, 1].set_ylabel("t-SNE 2", fontsize=20)
axes[0, 1].grid(False)
for spine in axes[0, 1].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
axes[0, 1].annotate(
    "B",
    xy=(-0.2, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)


logger.info("Calculating embeddings")
embeddings = _calc_embeddings(sdf=sdf_train, model=model, device="cpu")
embeddings = embeddings.numpy()
logger.info("Plotting")
tsne = TSNE(n_components=2)
tsne.fit(embeddings)
tsne_embeddings = tsne.fit_transform(embeddings)

sns.scatterplot(
    x=tsne_embeddings[:, 0],
    y=tsne_embeddings[:, 1],
    hue=sdf_train.labels,
    palette=colors,
    ax=axes[1, 0],
    alpha=0.5,
    legend=False,
)
axes[1, 0].set_xlabel("t-SNE 1", fontsize=20)
axes[1, 0].set_ylabel("t-SNE 2", fontsize=20)
axes[1, 0].grid(False)
for spine in axes[1, 0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
axes[1, 0].annotate(
    "C",
    xy=(-0.2, 0.9),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)
artists = []
for i, label in enumerate(labels):
    artists.append(
        plt.Line2D(
            (0, 1),
            (0, 0),
            color=colors[i],
            label=label,
            marker="o",
            linestyle="",
            markersize=10,
        )
    )
plt.legend(
    handles=artists,
    bbox_to_anchor=(2.5, 1),
    loc="center right",
    borderaxespad=0.0,
    fontsize=16,
    frameon=False,
)
# increase spacing between subplots
plt.subplots_adjust(wspace=0.3)
plt.savefig("tsne.png", dpi=300, bbox_inches="tight")

# ============================== UMAP ==============================
