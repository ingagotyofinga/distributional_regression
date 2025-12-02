from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import distance_matrix
import torch.nn.functional as F


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
    n_distributions: int,
    n_samples: int,
    dim: int,
    n_components: int = 3,
    mean_range = ((0, 3),(5,8)),      # or ((0,3), (5,8)) for two separated ranges
    cov_scale: float = 0.30,
    eps_noise: float = 0.05,
    seed: int | None = None,
    # piecewise control
    r_thresh: float | None = None,   # set to None to auto-pick from inner/outer medians
    alpha: float = 0.6,             # rotation angle θ_i = α * r_i
    kappa0: float = 0.0,            # shear base  k_i = κ0 + κ1 r_i
    kappa1: float = 0.25,
    beta: float = 0.4,              # translation a_i = β * m̄_i (set 0 to disable)
    device: str = "cpu",
):
    """
    2D only. Each μ_i is a GMM with C components.

    If `mean_range` is a single (low, high) tuple, all component means are drawn from that box.
    If `mean_range` is a pair of tuples ((low1, high1), (low2, high2)), we build a *bimodal* design:
        - First floor(n_distributions/2) μ_i use (low1, high1)
        - Remaining μ_i use (low2, high2)
      If r_thresh is None, it will be auto-chosen as the midpoint between the medians of the
      two groups' RMS radii r_i = W2(μ_i, δ0) ≈ sqrt(mean(||x||^2)).

    Piecewise map by W2 distance to δ0 (RMS radius r_i):
        if r_i < r_thresh:  ν_i = R(θ_i) μ_i + a_i + ε,   θ_i = α r_i
        else:               ν_i = S(k_i) μ_i + a_i + ε,   k_i = κ0 + κ1 r_i
    where R(θ) = [[cos,-sin],[sin,cos]] and S(k) = [[1,k],[0,1]]; ε ~ N(0, eps_noise^2 I).

    Returns tensors of shape (n_distributions, n_samples, 2).
    """
    assert dim == 2, "Only 2D supported."
    if seed is not None:
        torch.manual_seed(seed)

    # Detect whether we were given two disjoint ranges
    two_ranges = (
        isinstance(mean_range, (tuple, list))
        and len(mean_range) == 2
        and isinstance(mean_range[0], (tuple, list))
        and isinstance(mean_range[1], (tuple, list))
        and len(mean_range[0]) == 2
        and len(mean_range[1]) == 2
        and not isinstance(mean_range[0][0], (tuple, list))
    )

    if two_ranges:
        (low1, high1), (low2, high2) = mean_range
        n_inner = n_distributions // 2
        groups = ["inner"] * n_inner + ["outer"] * (n_distributions - n_inner)
    else:
        low, high = mean_range
        groups = ["single"] * n_distributions

    mu_all = torch.empty(n_distributions, n_samples, dim, device=device)
    nu_all = torch.empty_like(mu_all)

    # --- First pass: sample μ_i and compute r_i (so we can auto-pick r_thresh if requested) ---
    r_vals = torch.empty(n_distributions, device=device)
    for i, grp in enumerate(groups):
        if grp == "inner":
            low, high = low1, high1
        elif grp == "outer":
            low, high = low2, high2
        else:
            # single-range mode
            pass  # low, high already set above if not two_ranges

        weights = torch.distributions.Dirichlet(torch.ones(n_components, device=device)).sample()
        comp_means = (high - low) * torch.rand(n_components, dim, device=device) + low
        comp_idx = torch.multinomial(weights, n_samples, replacement=True)

        mu_i = torch.randn(n_samples, dim, device=device) * cov_scale + comp_means[comp_idx]
        mu_all[i] = mu_i

        r_i = torch.sqrt((mu_i.pow(2).sum(dim=1)).mean())  # RMS radius ≈ W2(μ_i, δ0)
        r_vals[i] = r_i

    # Auto threshold (only meaningful if two separated groups)
    if r_thresh is None and two_ranges:
        inner_r = r_vals[: (n_distributions // 2)]
        outer_r = r_vals[(n_distributions // 2):]
        r_thresh = float((inner_r.median() + outer_r.median()) / 2.0)
    elif r_thresh is None:
        r_thresh = float(r_vals.median())

    # --- Second pass: apply the piecewise map to get ν_i ---
    for i in range(n_distributions):
        mu_i = mu_all[i]
        mbar = mu_i.mean(0)
        r_i = r_vals[i]

        if r_i < r_thresh:
            theta = alpha * r_i
            c, s = torch.cos(theta), torch.sin(theta)
            A = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])   # 2x2
        else:
            k = kappa0 + kappa1 * r_i
            A = torch.tensor([[1.0, k], [0.0, 1.0]], device=device)        # 2x2 shear

        a = beta * mbar
        T_mu = mu_i @ A.T + a
        eps = eps_noise * torch.randn(n_samples, dim, device=device)
        nu_i = T_mu + eps

        nu_all[i] = nu_i

    return mu_all, nu_all

def generate_5d_gmm_pairs(
    n_distributions: int,
    n_samples: int,
    dim: int = 5,
    n_components: int = 3,
    mean_range = ((0, 3),(5,8)),      # accepts scalars or length-dim vectors; may also be a pair of such boxes
    cov_scale: float = 0.30,
    eps_noise: float = 0.05,
    seed: int | None = None,
    # piecewise control
    r_thresh: float | None = None,   # None → auto from median(s)
    alpha: float = 0.6,              # rotation angle θ_i = α * r_i (inner)
    kappa0: float = 0.0,             # shear base (outer)
    kappa1: float = 0.25,            # shear slope: k_i = κ0 + κ1 r_i
    beta: float = 0.4,               # translation a_i = β * m̄_i
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Empirical GMM pairs in R^dim (default dim=5).

    Design:
      • If `mean_range` is (low, high): a single box. `low, high` can be scalars or length-dim sequences.
      • If `mean_range` is ((low1, high1), (low2, high2)): bimodal design
          - first floor(n_distributions/2) use box1, remainder use box2.
        If r_thresh is None, auto-pick as midpoint between the medians of the two groups' RMS radii
        r_i := sqrt(mean(||x||^2)) (≈ W2(μ_i, δ0)).

    Piecewise map (depends on RMS radius r_i):
        if r_i < r_thresh:
            ν_i = A_rot(θ_i) μ_i + a_i + ε, where θ_i = α r_i and A_rot is block Givens rotations on (1,2),(3,4),...
        else:
            ν_i = A_shear(k_i) μ_i + a_i + ε, where k_i = κ0 + κ1 r_i and A_shear applies x_{2p-1}←x_{2p-1}+k x_{2p} on pairs.

    Returns:
        mu_all, nu_all  with shape (n_distributions, n_samples, dim)
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)

    def _as_vec(x, dim):
        # scalar → (dim,), list/tuple/torch → tensor(dim,)
        if isinstance(x, (int, float)):
            return torch.full((dim,), float(x), device=dev, dtype=dtype)
        t = torch.as_tensor(x, device=dev, dtype=dtype)
        if t.numel() == 1:
            return t.repeat(dim)
        assert t.numel() == dim, f"Expected length-{dim} vector, got shape {tuple(t.shape)}"
        return t

    # Detect whether two boxes were provided
    two_ranges = (
        isinstance(mean_range, (tuple, list))
        and len(mean_range) == 2
        and isinstance(mean_range[0], (tuple, list))
        and isinstance(mean_range[1], (tuple, list))
        and len(mean_range[0]) == 2
        and len(mean_range[1]) == 2
    )

    if two_ranges:
        (low1_raw, high1_raw), (low2_raw, high2_raw) = mean_range
        low1, high1 = _as_vec(low1_raw, dim), _as_vec(high1_raw, dim)
        low2, high2 = _as_vec(low2_raw, dim), _as_vec(high2_raw, dim)
        n_inner = n_distributions // 2
        groups = ["inner"] * n_inner + ["outer"] * (n_distributions - n_inner)
    else:
        low_raw, high_raw = mean_range
        low, high = _as_vec(low_raw, dim), _as_vec(high_raw, dim)
        groups = ["single"] * n_distributions

    mu_all = torch.empty(n_distributions, n_samples, dim, device=dev, dtype=dtype)
    nu_all = torch.empty_like(mu_all)
    r_vals = torch.empty(n_distributions, device=dev, dtype=dtype)

    # --- First pass: sample μ_i and compute r_i ---
    for i, grp in enumerate(groups):
        if grp == "inner":
            lvec, hvec = low1, high1
        elif grp == "outer":
            lvec, hvec = low2, high2
        else:
            lvec, hvec = low, high

        # component weights + means
        weights = torch.distributions.Dirichlet(torch.ones(n_components, device=dev, dtype=dtype)).sample()
        comp_means = lvec + (hvec - lvec) * torch.rand(n_components, dim, device=dev, dtype=dtype)
        comp_idx = torch.multinomial(weights, n_samples, replacement=True)

        # isotropic comp covariances with scale cov_scale
        mu_i = torch.randn(n_samples, dim, device=dev, dtype=dtype) * cov_scale + comp_means[comp_idx]
        mu_all[i] = mu_i

        # RMS radius (≈ W2(μ_i, δ0))
        r_vals[i] = torch.sqrt((mu_i.pow(2).sum(dim=1)).mean())

    # Auto-threshold for piecewise split
    if r_thresh is None and two_ranges:
        inner_r = r_vals[: (n_distributions // 2)]
        outer_r = r_vals[(n_distributions // 2):]
        r_thresh = float((inner_r.median() + outer_r.median()) / 2.0)
    elif r_thresh is None:
        r_thresh = float(r_vals.median())

    # --- Helpers to build transforms in R^dim ---
    def block_rotation_matrix(theta: torch.Tensor):
        """
        Block Givens rotations over pairs (1,2), (3,4), ..., leave last dim if odd.
        theta: scalar tensor
        """
        A = torch.eye(dim, device=dev, dtype=dtype)
        c, s = torch.cos(theta), torch.sin(theta)
        for p in range(0, dim - 1, 2):
            A[p, p] = c;   A[p, p + 1] = -s
            A[p + 1, p] = s; A[p + 1, p + 1] = c
        return A

    def block_shear_matrix(k: torch.Tensor):
        """
        Shear on pairs: x_{2p-1} ← x_{2p-1} + k x_{2p}; leave last dim if odd.
        """
        A = torch.eye(dim, device=dev, dtype=dtype)
        for p in range(0, dim - 1, 2):
            A[p, p + 1] = k
        return A

    # --- Second pass: apply piecewise map to get ν_i ---
    for i in range(n_distributions):
        mu_i = mu_all[i]
        mbar = mu_i.mean(0)
        r_i = r_vals[i]

        if r_i < r_thresh:
            theta = torch.as_tensor(alpha, device=dev, dtype=dtype) * r_i
            A = block_rotation_matrix(theta)
        else:
            k = torch.as_tensor(kappa0, device=dev, dtype=dtype) + torch.as_tensor(kappa1, device=dev, dtype=dtype) * r_i
            A = block_shear_matrix(k)

        a = beta * mbar
        T_mu = mu_i @ A.T + a
        eps = eps_noise * torch.randn(n_samples, dim, device=dev, dtype=dtype)
        nu_all[i] = T_mu + eps

    return mu_all, nu_all


