# src/utils/kernel.py
import torch
from geomloss import SamplesLoss

# This blur is ONLY for computing a neighbor "distance"; pick something stable.
_sinkhorn_for_nn = SamplesLoss("sinkhorn", p=2, blur=0.06)

def knn_bandwidth(mu_0, training_measures, k=10):
    """
    Returns a LENGTH scale h ≈ sqrt( (squared) distance to the k-th NN ),
    suitable for a Gaussian weight exp( - d^2 / (2 h^2) ).
    """
    mu_0 = mu_0.squeeze(0)  # (k,2)

    # collect squared-like distances d2_i
    d2 = []
    with torch.no_grad():
        for mu_i in training_measures:  # (k,2)
            d2.append(float(_sinkhorn_for_nn(mu_0, mu_i)))  # ≈ length^2

    d2.sort()
    idx = max(0, min(k-1, len(d2)-1))   # k-th NN (1-based) -> index k-1
    h = (d2[idx] ** 0.5)                # convert squared scale -> length
    return h
