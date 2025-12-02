from src.loss import gaussian_kernel
import numpy as np

def sample_training_set(metadata, mu0, k_train, n_train, bandwidth_k=5, kernel_thresh=0.05):
    # Lazy version: skip kernel checks and just sample at random
    total = metadata['mu'].shape[0]
    chosen_idxs = np.random.choice(total, size=n_train, replace=False)
    mu_train = metadata['mu'][chosen_idxs, :k_train]
    nu_train = metadata['nu'][chosen_idxs, :k_train]
    return mu_train, nu_train, chosen_idxs
