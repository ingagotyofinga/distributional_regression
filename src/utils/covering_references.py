import torch
from pathlib import Path
from src.data_utils.pair_dataset import DistributionPairDataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

torch.manual_seed(0)

def select_covering_references(mu_set, h=0.2, C=0.05, min_coverage=4):
    r = h * np.sqrt(-np.log(C))  # coverage radius
    n = mu_set.shape[0]
    means = mu_set.mean(dim=1)  # shape (n, d)

    # Precompute pairwise distances
    dists = torch.cdist(means, means)  # shape (n, n)

    uncovered = set(range(n))
    selected_refs = []
    all_coverage = []

    while uncovered:
        best_idx = None
        best_covered = set()

        for i in uncovered:
            neighbors = {j for j in uncovered if dists[i, j] <= r}
            if len(neighbors) >= min_coverage and len(neighbors) > len(best_covered):
                best_covered = neighbors
                best_idx = i

        if best_idx is None:
            break

        selected_refs.append(best_idx)
        all_coverage.append(sorted(best_covered))
        uncovered -= best_covered

    return selected_refs, all_coverage, r

# # === Load data ===
# path_to_data = Path("data/generated/pairs_n100_k1000_d2.pt")
# dataset = DistributionPairDataset(path_to_data)
# train_dataset, _ = dataset.train_val_split()
# train_indices = train_dataset.indices
# mu = dataset.mu[train_indices]  # shape: (n_train, k, d)
#
# # === Run covering selection ===
# h = 0.4
# C = 0.05
# refs, coverage, r = select_covering_references(mu, h=h, C=C, min_coverage=4)
#
# print(f"Selected {len(refs)} reference measures with blur={h}")
# for j, idx in enumerate(refs):
#     print(f"  mu[{idx}] covers {len(coverage[j])} distributions")
#
# # === Visualization ===
# if mu.shape[2] == 2:
#     means = mu.mean(dim=1)
#     fig, ax = plt.subplots(figsize=(6, 6))
#     colors = plt.cm.tab20.colors
#
#     for i, group in enumerate(coverage):
#         group_means = means[list(group)]
#         ax.scatter(group_means[:, 0], group_means[:, 1], s=10, color=colors[i % 20], alpha=0.6, label=f"ref {i}")
#
#     ref_means = means[refs]
#     ax.scatter(ref_means[:, 0], ref_means[:, 1], s=80, c='black', marker='x', label='references')
#
#     # Draw circles of radius r around each reference mean
#     for x, y in ref_means:
#         circle = Circle((x.item(), y.item()), radius=r, edgecolor='black', facecolor='none', linestyle='--', linewidth=1)
#         ax.add_patch(circle)
#
#     # Highlight uncovered points
#     covered_points = set(j for group in coverage for j in group)
#     all_points = set(range(means.shape[0]))
#     uncovered_points = all_points - covered_points
#
#     if uncovered_points:
#         unc_means = means[list(uncovered_points)]
#         ax.scatter(unc_means[:, 0], unc_means[:, 1], s=30, facecolors='none', edgecolors='red', label='uncovered')
#
#     ax.set_title(f"Coverage with blur={h}")
#     ax.set_aspect('equal')
#     # ax.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.25, 1))
#     plt.tight_layout()
#     plt.show()
