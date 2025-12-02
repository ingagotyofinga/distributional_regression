import torch
from geomloss import SamplesLoss
from tqdm import tqdm
import numpy as np

def compute_w2_stats(distributions, blur=0.5, p=2):
    """
    Args:
        distributions: list of tensors, each of shape (n_samples, dim)
        blur: Sinkhorn blur parameter
        p: power for W_p (typically 2)
    Returns:
        mean_W2, median_W2
    """
    sinkhorn_loss = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=0.9)

    pairwise_w2s = []
    n = len(distributions)

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            w2 = sinkhorn_loss(distributions[i], distributions[j]).item()
            pairwise_w2s.append(w2)

    pairwise_w2s = np.array(pairwise_w2s)
    return np.mean(pairwise_w2s), np.median(pairwise_w2s)
