import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import seaborn as sns
import subprocess
import os
from tqdm import tqdm
import pickle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 20

output_path = "reg.png"

kegg_fasta_path = "/davidb/bio_db/Kegg/2023-05-24/FAA/"
subcluster_fastas_path = "data/kegg/data_dir/subclusters"
all_sequences_path = "/davidb/bio_db/Kegg/2023-05-24/all_prok_euk_viruses.fasta"

if os.path.exists("ko_subcluster_size_distribution_data.pkl"):
    with open("ko_subcluster_size_distribution_data.pkl", "rb") as f:
        (
            n_lines_list_kos,
            n_lines_list_sc,
            n_subclusters,
            total_ko_sequences,
            total_sc_sequences,
            total_sequences,
            ko_to_number_of_subclusters,
            ko_to_ko_size,
        ) = pickle.load(f)
else:
    
    total_sequences = subprocess.check_output(["grep", "-c", ">", all_sequences_path])
    total_sequences = int(total_sequences.decode("utf-8").split()[0])
    # KO size distribution
    ko_list_files_paths = [
        os.path.join(kegg_fasta_path, f)
        for f in os.listdir(kegg_fasta_path)
        if f.endswith(".list")
        and os.path.getsize(
            os.path.join(kegg_fasta_path, f.removesuffix("list") + "faa")
        )
        > 0
    ]

    n_lines_list_kos = []
    ko_to_ko_size = {}
    for path in tqdm(ko_list_files_paths, desc="Counting lines in KO list files"):
        n_lines = subprocess.check_output(["wc", "-l", path]).decode("utf-8")
        n_lines = int(n_lines.split()[0])
        n_lines_list_kos.append(n_lines)
        ko = os.path.basename(path).split(".")[0]
        ko_to_ko_size[ko] = n_lines

    n_lines_list_kos = np.array(n_lines_list_kos)
    n_lines_list_kos = n_lines_list_kos[n_lines_list_kos > 0]
    total_ko_sequences = n_lines_list_kos.sum()

    # subcluster size distribution and number of subclusters
    subcluster_fasta_file_paths = [
        os.path.join(subcluster_fastas_path, f)
        for f in os.listdir(subcluster_fastas_path)
        if f.endswith(".fasta")
    ]

    n_lines_list_sc = []
    ko_to_number_of_subclusters = {}

    for path in tqdm(
        subcluster_fasta_file_paths, desc="Counting lines in subcluster fasta files"
    ):
        ko = os.path.basename(path).split(".")[0]
        if ko not in ko_to_number_of_subclusters:
            ko_to_number_of_subclusters[ko] = 0
        ko_to_number_of_subclusters[ko] += 1
        n_lines = subprocess.check_output(["grep", "-c", ">", path]).decode("utf-8")
        n_lines = int(n_lines.split()[0])
        n_lines_list_sc.append(n_lines)

    n_lines_list_sc = np.array(n_lines_list_sc)
    n_lines_list_sc = n_lines_list_sc[n_lines_list_sc > 0]
    total_sc_sequences = n_lines_list_sc.sum()
    n_subclusters = np.array(list(ko_to_number_of_subclusters.values()))

    # save data
    data = (
        n_lines_list_kos,
        n_lines_list_sc,
        n_subclusters,
        total_ko_sequences,
        total_sc_sequences,
        total_sequences,
        ko_to_number_of_subclusters,
        ko_to_ko_size,
    )
    with open("ko_subcluster_size_distribution_data.pkl", "wb") as f:
        pickle.dump(data, f)
# Plotting
# subplot 1: all sequences, all KOs sequences, all subcluster sequences bar plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.set_style("whitegrid")
ax: plt.Axes = axes[0]
print(f"Total sequences: {total_sequences}")
print(f"Total KO sequences: {total_ko_sequences}")
print(f"Total subcluster sequences: {total_sc_sequences}")
sns.set_palette("pastel")
palette = [
        sns.color_palette()[0],
        sns.color_palette()[1],
        sns.color_palette()[2],
    ]
sns.barplot(
    x=["Subclusrers only", "All labeled", "Total"],
    y=[total_sc_sequences, total_ko_sequences, total_sequences],
    ax=ax,
    palette=palette,
    linewidth=1,
    edgecolor="black",
)
ax.set_xticklabels(
    ["Subclusrers only", "All labeled", "Total"], rotation=30, ha="right"
)
ax.text(
    x=-0.125,
    y=1,
    s='A',
    fontsize=30,
    ha='right',
    va='top',
    transform=ax.transAxes
)
    

num_ticks = total_sequences // 5000000 + 1
y_ticks = np.arange(0, 5000000 * (num_ticks), 5000000)
y_tick_labels = [str(5 * i) for i in range(num_ticks)]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)
ax.set_ylabel("Number of sequences (millions)", labelpad=10)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
# ax.set_title("Log number of sequences")
# subplot 2: KO size distribution kde
ax2: plt.Axes = axes[1]
log_n_lines_list_kos = np.log10(n_lines_list_kos)
# make shade color same as outline color
sns.kdeplot(x=log_n_lines_list_kos, ax=ax2, shade=True, color=palette[1],
            linewidth=1, alpha=1, edgecolor="black")
x_max_rounded = int(log_n_lines_list_kos.max())
# xticks = [1]
# xticklables = ["$10^0$"]
# for i in range(x_max_rounded):
#     tick = 10**i
#     step = 10**i
#     for j in range(1, 11):
#         if j == 10:
#             xticklables.append(f"$10^{i + 1}$")
#         else:
#             xticklables.append("")
#         xticks.append(tick + j * step)

xticks = [1]
xticklables = ["$10^0$"]
for i in range(x_max_rounded):
    tick = 10**i
    step = 10**i
    xticklables.append(f"$10^{i + 1}$")
    xticks.append(tick + 10 * step)


xticks = np.log10(xticks)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklables)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
# ax2.set_title("KO size distribution")
ax2.set_xlabel("KO size")
ax2.set_ylabel("Density", labelpad=10)
ax2.grid(False)
xticks = ax2.get_xticks()
xticklabels = ax2.get_xticklabels()
ax2.text(
    x=-0.125,
    y=1,
    s='B',
    fontsize=30,
    ha='right',
    va='top',
    transform=ax2.transAxes
)
plt.savefig("bar_and_ko_sizes.png", dpi=300, bbox_inches="tight")

ko_to_ko_size = {
    k: v for k, v in ko_to_ko_size.items() if k in ko_to_number_of_subclusters
}
kos = np.array(list(ko_to_ko_size.keys()))
ko_sizes = np.array([ko_to_ko_size[ko] for ko in kos])
ko_sizes = np.log10(ko_sizes)
ko_n_subclusters = np.array([ko_to_number_of_subclusters[ko] for ko in kos])
ko_n_subclusters = np.log10(ko_n_subclusters)

plt.suptitle("KO size vs number of subclusters")
plt.savefig("scatter.png", dpi=300, bbox_inches="tight")
# subplot 3: KO size vs number of subclusters kde
jp: sns.JointGrid = sns.jointplot(
    x=ko_sizes,
    y=ko_n_subclusters,
    kind="kde",
    fill=True,
    # levels=[0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    levels=10,
    height=8,
)
jp.ax_joint.set_xlabel("N sequences in KO")
jp.ax_joint.set_ylabel("N subclusters in KO")
x_max_rounded = int(ko_sizes.max())
y_max_rounded = int(ko_n_subclusters.max())
# jp.ax_joint.set_xticks(
#     np.arange(0, x_max_rounded + 1, 1),
# )
# xticks = [1]
# xticklables = ["$10^0$"]
# for i in range(x_max_rounded):
#     tick = 10**i
#     step = 10**i
#     for j in range(1, 11):
#         if j == 10:
#             xticklables.append(f"$10^{i + 1}$")
#         else:
#             xticklables.append("")
#         xticks.append(tick + j * step)
# xticks = np.log10(xticks)
# jp.ax_joint.set_xticks(xticks)
# jp.ax_joint.set_xticklabels(xticklables)
# yticks = [1]
# yticklables = ["$10^0$"]
# for i in range(y_max_rounded):
#     tick = 10**i
#     step = 10**i
#     for j in range(1, 11):
#         if j == 10:
#             yticklables.append(f"$10^{i + 1}$")
#         else:
#             yticklables.append("")
#         yticks.append(tick + j * step)
# yticks = np.log10(yticks)
# jp.ax_joint.set_yticks(yticks)
# jp.ax_joint.set_yticklabels(yticklables)


jp.ax_joint.tick_params(axis="both", which="major", reset=True, top=False, right=False)
for spine in jp.ax_joint.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_color("black")
jp.ax_joint.grid(False)
# remove grid from marginal plots
jp.ax_marg_x.grid(False)
jp.ax_marg_y.grid(False)
jp.ax_joint.set_xticks(xticks)
jp.ax_joint.set_xticklabels(xticklables)
yticks = [
    1,
    5,
    10,
    50,
    100,
]
yticklables = [
    1,
    5,
    10,
    50,
    100,
]
yticks = np.log10(yticks)
jp.ax_joint.set_yticks(yticks)
jp.ax_joint.set_yticklabels(yticklables)
jp.ax_joint.set_ylim(-0.3, yticks[-1])
# scatter plot over kde
jp.ax_joint.scatter(
    ko_sizes,
    ko_n_subclusters,
    marker="o",
    color="black",
    alpha=0.2,
    s=6,
)

# increase space between plots
jp.fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.savefig("kde.png", dpi=300, bbox_inches="tight")
