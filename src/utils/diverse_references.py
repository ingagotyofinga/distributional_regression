import torch

def select_diverse_mu0s(mu_set: torch.Tensor, k: int) -> list:
    """
    Selects k diverse reference indices from mu_set using greedy farthest-point sampling on means.

    Args:
        mu_set (Tensor): Tensor of shape (n_dists, n_samples, dim)
        k (int): Number of reference distributions to select

    Returns:
        List[int]: Indices of selected reference distributions
    """
    n_dists, n_samples, dim = mu_set.shape
    means = mu_set.mean(dim=1)  # (n_dists, dim)

    selected = [0]  # start with the first
    while len(selected) < k:
        remaining = list(set(range(n_dists)) - set(selected))
        dists = []

        for idx in remaining:
            min_dist = min(
                torch.norm(means[idx] - means[s]).item()
                for s in selected
            )
            dists.append((min_dist, idx))

        # pick the one with the maximum min-distance
        _, next_idx = max(dists)
        selected.append(next_idx)

    return selected
