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
import umap
import random
import matplotlib
from app.model import load_from_state

matplotlib.use("Agg")

sns.set_style("ticks")


def l2_normalize(array):
    # Handle both 1D and 2D arrays
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        return array / norm if norm > 0 else array
    elif array.ndim == 2:
        # Normalize each row independently
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return np.where(norms > 0, array / norms, array)
    else:
        raise ValueError("Input must be a 1D or 2D NumPy array")


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
    "K01886",
    "K01893",
    "K01878",
    "K01879",
    "K01889",
    "K01890",
    "K01892",
    "K01881",
    "K04566",
    "K04567",
    "K01883",
    "K01885",
    "K01887",
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
    "Glutaminyl-tRNA synthetase",
    "Asparaginyl-tRNA synthetase",
    "Glycyl-tRNA synthetase alpha chain",
    "Glycyl-tRNA synthetase beta chain",
    "Phenylalanyl-tRNA synthetase alpha chain",
    "Phenylalanyl-tRNA synthetase beta chain",
    "Histidyl-tRNA synthetase",
    "Prolyl-tRNA synthetase",
    "Lysyl-tRNA synthetase, class I",
    "Lysyl-tRNA synthetase, class II",
    "Cysteinyl-tRNA synthetase",
    "Glutamyl-tRNA synthetase",
    "Arginyl-tRNA synthetase",
]

ko_to_label = dict(zip(kos, labels))

colors = (
    list(sns.color_palette("pastel", n_colors=10))
    + list(sns.color_palette("dark", n_colors=10))
    + list(sns.color_palette("bright", n_colors=10))
)
random.shuffle(colors)
colors = colors[: len(kos)]

print("Started")

sdf_train_path = "models/full/kegg_2021_dedup/data_dir/sdf_train.pkl"
sdf_test_path = "data/all_synthetases_full/kegg_2021_dedup/sdf_classify.pkl"
model_state_path = (
    "models/full/kegg_2021_dedup/checkpoints/epoch_5/1470384_checkpoint.pt"
)
gt = "/davidb/guyshur/kegg_data/2023/ground_truth.pkl"
logger.info("Loading model")
print("Loading model")
model = load_from_state(model_state_path, device=torch.device("cpu"))
model.eval()
print("Loading data")

sdf_test: SparseDataFrame = pickle.load(open(sdf_test_path, "rb"))

sdf_test_index_ids = list(sdf_test.index_ids)


print("Loading ground truth")
gt = pickle.load(open(gt, "rb"))
sdf_test.labels = {
    k: gt[k] if type(gt[k]) is list else [gt[k]] for k in sdf_test.index_ids
}

sdf_test.labels = np.array([sdf_test.labels[k][0] for k in sdf_test.index_ids])
sdf_test.labels = np.array([ko_to_label[k] for k in sdf_test.labels])

test_raw_data = sdf_test.matrix.toarray()
import os
import pickle

# if not os.path.exists("umap_light_raw.pkl"):
print("Calculating umap")
manifold = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.01)

test_manifold_data = manifold.fit_transform(test_raw_data)
#     with open("umap_light_raw.pkl", "wb") as f:
#         pickle.dump(test_manifold_data, f)
# else:
#     with open("umap_light_raw.pkl", "rb") as f:
#         test_manifold_data = pickle.load(f)
print("Plotting")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 9))

sns.scatterplot(
    x=test_manifold_data[:, 0],
    y=test_manifold_data[:, 1],
    hue=sdf_test.labels,
    palette=colors,
    ax=axes[0],
    s=10,
    alpha=0.5,
    legend=False,
    hue_order=labels,
)

axes[0].set_xlabel("UMAP 1", fontsize=10)
axes[0].set_ylabel("UMAP 2", fontsize=10)
axes[0].grid(False)
for spine in axes[0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
axes[0].annotate(
    "A",
    xy=(-0.25, 0.92),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)

print("Normalizing data")
test_norm_data = l2_normalize(test_raw_data)
# if not os.path.exists("umap_light_norm.pkl"):
print("Calculating umap")
test_manifold_norm = manifold.fit_transform(test_norm_data)
#    with open("umap_light_norm.pkl", "wb") as f:
#        pickle.dump(test_manifold_norm, f)
# else:
#    with open("umap_light_norm.pkl", "rb") as f:
#        test_manifold_norm = pickle.load(f)
print("Plotting")

sns.scatterplot(
    x=test_manifold_norm[:, 0],
    y=test_manifold_norm[:, 1],
    hue=sdf_test.labels,
    palette=colors,
    ax=axes[1],
    s=10,
    alpha=0.5,
    legend=False,
    hue_order=labels,
)
axes[1].set_xlabel("UMAP 1", fontsize=10)
axes[1].set_ylabel("UMAP 2", fontsize=10)
axes[1].grid(False)
for spine in axes[1].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")

axes[1].annotate(
    "B",
    xy=(-0.25, 0.92),
    xycoords="axes fraction",
    fontsize=16,
    horizontalalignment="left",
    verticalalignment="bottom",
)
# if not os.path.exists("umap_light_mf.pkl"):
logger.info("Calculating embeddings")
test_embeddings = _calc_embeddings(sdf=sdf_test, model=model, device="cpu")
#     test_embeddings = test_embeddings
print("Calculating embeddings manifold")
test_manifold_embeddings = manifold.fit_transform(test_embeddings)
#     with open("umap_light_mf.pkl", "wb") as f:
#         pickle.dump(test_manifold_embeddings, f)
# else:
#     with open("umap_light_mf.pkl", "rb") as f:
#         test_manifold_embeddings = pickle.load(f)

print("Plotting")

sns.scatterplot(
    x=test_manifold_embeddings[:, 0],
    y=test_manifold_embeddings[:, 1],
    hue=sdf_test.labels,
    palette=colors,
    ax=axes[2],
    s=10,
    alpha=0.5,
    legend=False,
    hue_order=labels,
)
axes[2].set_xlabel("UMAP 1", fontsize=10)
axes[2].set_ylabel("UMAP 2", fontsize=10)
axes[2].grid(False)
for spine in axes[2].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
axes[2].annotate(
    "C",
    xy=(-0.25, 0.92),
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
            markersize=9,
        )
    )
axes[1].legend(
    handles=artists,
    bbox_to_anchor=(0.95, 0.5),
    labelspacing=1,
    loc="center left",
    borderaxespad=0.0,
    fontsize=8,
    frameon=False,
    ncols=1,
    bbox_transform=fig.transFigure,
)
for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.tick_params(axis="both", which="minor", labelsize=6)
plt.savefig("umap_20.png", dpi=300, bbox_inches="tight")
