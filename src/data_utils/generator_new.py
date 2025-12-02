# src/data_utils/generator.py

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# -----------------------------
# 2D single-Gaussian pairs
# -----------------------------
def generate_gaussian_pairs(
    n_distributions: int,
    n_samples: int,
    dim: int,
    mean_range=(-3, 3),
    eps_noise: float = 0.05,
    seed: int | None = None,
):
    """
    2D only. Source μ_i ~ N(m_i, I), with m_i ∈ [mean_range]^2.
    Target ν_i = A_i μ_i + a_i + ε, where A_i is a rotation depending on ||m_i||,
    and a_i = 0.5 m_i. ε ~ N(0, eps_noise^2 I).
    Returns tensors of shape (n_distributions, n_samples, 2).
    """
    assert dim == 2, "generate_gaussian_pairs supports 2D only."
    if seed is not None:
        torch.manual_seed(seed)

    mu_list, nu_list = [], []

    for _ in range(n_distributions):
        # mean in box
        m_i = (mean_range[1] - mean_range[0]) * torch.rand(dim) + mean_range[0]
        # source cloud
        mu_i = torch.randn(n_samples, dim) + m_i

        # rotation based on ||m_i||
        theta = 2 * torch.norm(m_i)
        c, s = torch.cos(theta), torch.sin(theta)
        A_i = torch.tensor([[c, -s],
                            [s,  c]], dtype=mu_i.dtype, device=mu_i.device)

        # translation
        a_i = 0.5 * m_i

        # map + small noise
        T_mu_i = mu_i @ A_i.T + a_i
        nu_i = T_mu_i + eps_noise * torch.randn_like(T_mu_i)

        mu_list.append(mu_i)
        nu_list.append(nu_i)

    mu_tensor = torch.stack(mu_list)  # (n, k, 2)
    nu_tensor = torch.stack(nu_list)
    return mu_tensor, nu_tensor


# -----------------------------
# 2D GMM pairs
# -----------------------------
def generate_2d_gmm_pairs(
    n_distributions: int,
    n_samples: int,
    dim: int,
    n_components: int = 3,
    mean_range = (-5, 5),
    cov_scale: float = 0.3,
    eps_noise: float = 0.05,
    seed: int | None = None,
    map_alpha: float = 2.0,   # controls rotation magnitude θ = α‖m̄‖
    map_beta: float = 0.5,    # controls translation a = β m̄
):
    """
    2D only. Source μ_i is a GMM with C components (isotropic cov cov_scale^2 I).
    Map varies per distribution using its barycenter m̄:
        θ_i = α * ||m̄_i||,  a_i = β * m̄_i,
        ν_i = R(θ_i) μ_i + a_i + ε.
    Returns tensors of shape (n_distributions, n_samples, 2).
    """
    assert dim == 2, "generate_2d_gmm_pairs supports 2D only."
    if seed is not None:
        torch.manual_seed(seed)

    mu_list, nu_list = [], []

    for _ in range(n_distributions):
        # Dirichlet weights and component means
        weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample()              # (C,)
        comp_means = (mean_range[1] - mean_range[0]) * torch.rand(n_components, dim) + mean_range[0]  # (C,2)

        # Weighted (population) barycenter m̄ (more stable than sample mean for small k)
        m_bar = (weights.unsqueeze(1) * comp_means).sum(0)   # (2,)

        # Sample exactly n_samples points from the mixture
        comp_idx = torch.multinomial(weights, num_samples=n_samples, replacement=True)         # (n,)
        z = torch.randn(n_samples, dim) * cov_scale
        mu_i = z + comp_means[comp_idx]  # (n,2)

        # Per-distribution map: rotation depends on ||m̄||, translation on m̄
        theta = map_alpha * torch.norm(m_bar)               # scalar
        c, s = torch.cos(theta), torch.sin(theta)
        A_i = torch.tensor([[c, -s],
                            [s,  c]], dtype=mu_i.dtype, device=mu_i.device)
        a_i = map_beta * m_bar                              # (2,)

        # Apply map + small noise
        T_mu_i = mu_i @ A_i.T + a_i
        nu_i = T_mu_i + eps_noise * torch.randn_like(T_mu_i)

        mu_list.append(mu_i)
        nu_list.append(nu_i)

    mu_tensor = torch.stack(mu_list)  # (n_distributions, n_samples, 2)
    nu_tensor = torch.stack(nu_list)
    return mu_tensor, nu_tensor



# -----------------------------
# 5D GMM pairs
# -----------------------------
def generate_5d_gmm_pairs(
    n_distributions: int,
    n_samples: int,
    dim: int,
    n_components: int = 3,
    mean_range = (-5, 5),
    eps_noise: float = 0.05,
    seed: int | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    Dimension-agnostic (used for d=5). Source μ_i is a GMM with C components.
    Target applies a smooth, input-dependent map:
        R(x) = exp(theta(x) * A_skew), theta(x) = alpha * tanh(||x||)
        b(x) = beta * tanh(Wx)
        ν_i = R(x) x + b(x) + ε
    Returns tensors of shape (n_distributions, n_samples, dim).
    """
    if seed is not None:
        torch.manual_seed(seed)

    mu_list, nu_list = [], []

    # fixed skew-symmetric matrix for all distributions
    M = torch.randn(dim, dim)
    skew_A = M - M.T  # (d, d)
    # simple structured W
    W = torch.full((dim, dim), -0.2)
    W.fill_diagonal_(0.5)

    for _ in tqdm(range(n_distributions), desc="Generating 5D GMM pairs"):
        # component means & weights
        comp_means = (mean_range[1] - mean_range[0]) * torch.rand(n_components, dim) + mean_range[0]
        weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample()

        # assignments (sum to n_samples)
        comp_idx = torch.multinomial(weights, num_samples=n_samples, replacement=True)

        # GMM samples
        z = torch.randn(n_samples, dim)
        mu_i = z + comp_means[comp_idx]  # (n, d)

        # theta per sample
        theta_vals = alpha * torch.tanh(mu_i.norm(dim=1))  # (n,)
        # rotation matrices via matrix exponential
        R = torch.stack([torch.matrix_exp(theta * skew_A) for theta in theta_vals])  # (n, d, d)
        # smooth shift
        b_vals = beta * torch.tanh(mu_i @ W.T)  # (n, d)

        # apply map + noise
        T_mu_i = torch.bmm(R, mu_i.unsqueeze(-1)).squeeze(-1) + b_vals
        nu_i = T_mu_i + eps_noise * torch.randn_like(T_mu_i)

        mu_list.append(mu_i)
        nu_list.append(nu_i)

    mu_tensor = torch.stack(mu_list)  # (n, k, d)
    nu_tensor = torch.stack(nu_list)
    return mu_tensor, nu_tensor

def _pca2(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project (k,d) tensors x,y -> (k,2) using a shared 2D PCA basis
    computed from the concatenation [x; y] (so both clouds share axes).
    """
    xy = torch.cat([x, y], dim=0)                      # (2k, d)
    xy0 = xy - xy.mean(dim=0, keepdim=True)            # center
    # SVD on centered data: xy0 ≈ U diag(S) Vh
    U, S, Vh = torch.linalg.svd(xy0, full_matrices=False)
    W = Vh[:2].T                                       # (d,2) top components
    return (x - xy.mean(0)) @ W, (y - xy.mean(0)) @ W  # project with shared basis

def preview_generated_pairs(
    mu_tensor: torch.Tensor,
    nu_tensor: torch.Tensor,
    dim: int,
    max_examples: int = 4,
    point_size: int = 6,
    alpha: float = 0.75,
    save_path: str | None = None,
):
    """
    Visualize up to `max_examples` (mu, nu) pairs from stacked tensors.
    If dim==2: raw scatter. If dim>2: PCA->2D with a shared basis per pair.
    """
    assert mu_tensor.shape == nu_tensor.shape and mu_tensor.ndim == 3
    n, k, d = mu_tensor.shape
    m = min(max_examples, n)

    fig, axes = plt.subplots(1, m, figsize=(4.2*m, 4.2), constrained_layout=True)
    if m == 1:
        axes = [axes]

    for i in range(m):
        mu_i = mu_tensor[i].detach().cpu()
        nu_i = nu_tensor[i].detach().cpu()

        if dim == 2:
            mu2, nu2 = mu_i, nu_i
        else:
            mu2, nu2 = _pca2(mu_i, nu_i)

        ax = axes[i]
        ax.scatter(mu2[:, 0], mu2[:, 1], s=point_size, alpha=alpha, label="μ (source)")
        ax.scatter(nu2[:, 0], nu2[:, 1], s=point_size, alpha=alpha, label="ν (target)")
        ax.set_title(f"Pair {i}")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_aspect("equal")
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved preview to {save_path}")
    else:
        plt.show()


# Example (uncomment to try right after generation):

mu, nu = generate_2d_gmm_pairs(n_distributions=3, n_samples=1000, dim=2, seed=42)
preview_generated_pairs(mu, nu, dim=2, max_examples=3)

# mu5, nu5 = generate_5d_gmm_pairs(n_distributions=3, n_samples=500, dim=5, seed=0)
# preview_generated_pairs(mu5, nu5, dim=5, max_examples=3, save_path="preview_5d.png")