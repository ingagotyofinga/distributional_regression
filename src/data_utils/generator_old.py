from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import distance_matrix


def generate_gaussian_pairs(
    n_distributions,
    n_samples,
    dim,
    mean_range=(-3, 3),
    eps_noise=0.05,
    seed=None
):
    assert dim == 2, "This version assumes 2D distributions for rotation-based A."

    if seed is not None:
        torch.manual_seed(seed)

    mu_list = []
    nu_list = []

    for _ in range(n_distributions):
        # Step 1: Choose mean m_i ∈ [low, high]^2
        m_i = (mean_range[1] - mean_range[0]) * torch.rand(dim) + mean_range[0]

        # Step 2: Sample mu_i ~ N(m_i, I)
        mu_i = torch.randn(n_samples, dim) + m_i  # shape: (n_samples, 2)

        # Step 3: Define a smooth rotation angle based on norm of m_i
        theta = 2 * torch.norm(m_i)  # scalar
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        A_i = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])  # shape: (2, 2)

        # Step 4: Translation vector proportional to m_i
        a_i = 0.5 * m_i

        # Apply the structured map: T_mu_i(x) = A_i x + a_i
        T_mu_i = mu_i @ A_i.T + a_i  # shape: (n_samples, 2)

        # Step 5: Add small random noise ε_i ~ N(0, σ^2 I)
        eps = eps_noise * torch.randn(dim)
        nu_i = T_mu_i + eps  # shape: (n_samples, 2)

        mu_list.append(mu_i)
        nu_list.append(nu_i)

    mu_tensor = torch.stack(mu_list)  # (n_distributions, n_samples, 2)
    nu_tensor = torch.stack(nu_list)

    return mu_tensor, nu_tensor

def generate_2d_gmm_pairs(
    n_distributions,
    n_samples,
    dim,
    n_components=3,
    mean_range=(-5, 5),
    cov_scale=0.3,
    eps_noise=0.05,
    seed=None
):
    assert dim == 2, "Only 2D GMMs are supported currently."
    if seed is not None:
        torch.manual_seed(seed)

    mu_list = []
    nu_list = []

    for _ in range(n_distributions):
        # Mixture weights
        weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample()

        # Component means
        comp_means = (mean_range[1] - mean_range[0]) * torch.rand(n_components, dim) + mean_range[0]

        # Component samples
        mu_i = []
        for j in range(n_components):
            n_j = int(n_samples * weights[j])
            samples = torch.randn(n_j, dim) * cov_scale + comp_means[j]
            mu_i.append(samples)
        mu_i = torch.cat(mu_i, dim=0)

        # Structured map to push them apart
        theta = torch.tensor(torch.pi / 2)  # 90 degree rotation
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        A_i = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        a_i = torch.tensor([4.0, 4.0])  # large translation

        T_mu_i = mu_i @ A_i.T + a_i
        eps = eps_noise * torch.randn(dim)
        nu_i = T_mu_i + eps

        mu_list.append(mu_i)
        nu_list.append(nu_i)

    max_len = max(mu.shape[0] for mu in mu_list)
    mu_tensor = torch.stack([torch.nn.functional.pad(mu, (0, 0, 0, max_len - mu.shape[0])) for mu in mu_list])
    nu_tensor = torch.stack([torch.nn.functional.pad(nu, (0, 0, 0, max_len - nu.shape[0])) for nu in nu_list])

    return mu_tensor, nu_tensor

def generate_5d_gmm_pairs(
    n_distributions,
    n_samples,
    dim,
    n_components=3,
    mean_range=(-5, 5),
    eps_noise=0.05,
    seed=None,
    alpha=1.0,
    beta=1.0,
):
    if seed is not None:
        torch.manual_seed(seed)

    mu_list = []
    nu_list = []
    all_mu_samples = []

    # Step 0: Construct skew-symmetric matrix A (fixed for all distributions)
    M = torch.randn(dim, dim)
    skew_A = M - M.T  # (d, d)

    for _ in tqdm(range(n_distributions), desc="Generating distributions"):
        # Step 1: Random component means (C x d)
        component_means = (mean_range[1] - mean_range[0]) * torch.rand(n_components, dim) + mean_range[0]

        # Step 2: Random component weights (C,)
        weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample()

        # Step 3: Sample component indices for all n_samples (n,)
        component_indices = torch.multinomial(weights, num_samples=n_samples, replacement=True)

        # Step 4: Fast sampling from GMM using vectorized operations
        z = torch.randn(n_samples, dim)  # standard normal noise
        selected_means = component_means[component_indices]  # (n, d)
        mu_i = z + selected_means  # (n, d)
        all_mu_samples.append(mu_i)

        # Step 5: Compute theta(x) = alpha * tanh(||x||)
        norm_mu = mu_i.norm(dim=1)  # (n,)
        theta_vals = alpha * torch.tanh(norm_mu)  # (n,)

        # Step 6: Construct rotation matrices R(x) = exp(theta(x) * A)
        R_matrices = torch.stack([torch.matrix_exp(theta * skew_A) for theta in theta_vals])  # (n, d, d)

        # Step 7: Define smooth shift b(x) = beta * tanh(Wx)
        W = torch.tensor([[0.5 if i == j else -0.2 for j in range(dim)] for i in range(dim)])
        b_vals = beta * torch.tanh(mu_i @ W.T)  # (n, d)

        # Step 8: Apply transformation T(x) = R(x) @ x + b(x)
        T_mu_i = torch.bmm(R_matrices, mu_i.unsqueeze(-1)).squeeze(-1) + b_vals  # (n, d)

        # Step 9: Add small noise
        eps = eps_noise * torch.randn_like(T_mu_i)
        nu_i = T_mu_i + eps  # (n, d)

        mu_list.append(mu_i)
        nu_list.append(nu_i)

    mu_tensor = torch.stack(mu_list)  # (n_distributions, n_samples, dim)
    nu_tensor = torch.stack(nu_list)

    # Visualization and density diagnostics (only in 2D)
    if dim == 2:
        all_mu = torch.cat(all_mu_samples, dim=0).numpy()

        # 2D histogram (density heatmap)
        plt.figure(figsize=(6, 5))
        plt.hist2d(all_mu[:, 0], all_mu[:, 1], bins=50, range=[[mean_range[0], mean_range[1]], [mean_range[0], mean_range[1]]], cmap="viridis")
        plt.colorbar(label="Density")
        plt.title("Heatmap of Source Distribution Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

    return mu_tensor, nu_tensor


