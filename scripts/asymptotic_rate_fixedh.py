import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)

# =========================
# Global settings
# =========================
dim = 2
theta = torch.tensor(torch.pi / 4, dtype=torch.float32)
s = 2.0
EPS = 1e-6
MAX_ITER = 200
LR = 1e-1

DTYPE = torch.float32
DEVICE = "cpu"

I_dim = torch.eye(dim, dtype=DTYPE, device=DEVICE)

# True affine map
A_true = s * torch.tensor([
    [torch.cos(theta), -torch.sin(theta)],
    [torch.sin(theta),  torch.cos(theta)]
], dtype=DTYPE, device=DEVICE)

b_true = torch.tensor([[5.0], [3.0]], dtype=DTYPE, device=DEVICE)

# Constant covariances in this experiment
Sigma_const = I_dim.clone()
Gamma_const = A_true @ Sigma_const @ A_true.T


# =========================
# SPD matrix square root
# =========================
def matrix_sqrt_spd(M):
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=EPS)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def wasserstein_distance_gaussian_single(mu1, Sigma1, mu2, Sigma2):
    mean_term = torch.sum((mu1 - mu2) ** 2)

    Sigma1 = 0.5 * (Sigma1 + Sigma1.T) + EPS * I_dim
    Sigma2 = 0.5 * (Sigma2 + Sigma2.T) + EPS * I_dim

    Sigma1_sqrt = matrix_sqrt_spd(Sigma1)
    middle = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt
    middle = 0.5 * (middle + middle.T) + EPS * I_dim
    middle_sqrt = matrix_sqrt_spd(middle)

    trace_term = torch.trace(Sigma1 + Sigma2 - 2.0 * middle_sqrt)
    return mean_term + trace_term


def wasserstein_kernel_batch(mu0, Sigma0, m, bandwidth):
    """
    m: (n, dim, 1)
    returns weights: (n,)
    """
    zero = torch.zeros((dim, 1), dtype=DTYPE, device=DEVICE)

    cov_term = wasserstein_distance_gaussian_single(
        zero, Sigma0, zero, Sigma_const
    )

    mean_terms = torch.sum((m - mu0.unsqueeze(0)) ** 2, dim=(1, 2))
    w2_sq = mean_terms + cov_term

    return torch.exp(-w2_sq / (2.0 * bandwidth ** 2))


def apply_B_to_batch(B, x):
    """
    B: (dim, dim)
    x: (n, dim, 1)
    returns: (n, dim, 1)
    """
    return torch.matmul(B.unsqueeze(0), x)


def compute_alpha_vectorized(B, m, q, mu0, Sigma0, bandwidth):
    weights = wasserstein_kernel_batch(mu0, Sigma0, m, bandwidth)   # (n,)
    residual_terms = -apply_B_to_batch(B, m) + q                    # (n, dim, 1)

    weighted_sum = torch.sum(residual_terms * weights[:, None, None], dim=0)
    return weighted_sum / (torch.sum(weights) + EPS)


def compute_loss_vectorized(B, alpha, m, q, mu0, Sigma0, bandwidth):
    """
    Uses that Sigma_i = I and Gamma_i = Gamma_const for all i.
    Still loops over covariance part only once.
    """
    weights = wasserstein_kernel_batch(mu0, Sigma0, m, bandwidth)   # (n,)

    m_trans = apply_B_to_batch(B, m) + alpha.unsqueeze(0)           # (n, dim, 1)
    mean_terms = torch.sum((m_trans - q) ** 2, dim=(1, 2))          # (n,)

    Sigma_trans = B @ Sigma_const @ B.T
    cov_term = wasserstein_distance_gaussian_single(
        torch.zeros((dim, 1), dtype=DTYPE, device=DEVICE),
        Sigma_trans,
        torch.zeros((dim, 1), dtype=DTYPE, device=DEVICE),
        Gamma_const
    )

    w2_vals = mean_terms + cov_term
    return torch.mean(w2_vals * weights)


def run_single_experiment_fast(n, bandwidth, max_iter=MAX_ITER, lr=LR):
    err = 1e-2 * torch.randn(n, dim, 1, dtype=DTYPE, device=DEVICE)
    m = torch.randn(n, dim, 1, dtype=DTYPE, device=DEVICE)
    q = apply_B_to_batch(A_true, m) + b_true.unsqueeze(0) + err

    mu0 = torch.randn(dim, 1, dtype=DTYPE, device=DEVICE)
    Sigma0 = I_dim.clone()

    B = (I_dim + 0.01 * torch.randn(dim, dim, dtype=DTYPE, device=DEVICE)).detach()
    B.requires_grad_(True)

    optimizer = optim.Adam([B], lr=lr)

    alpha = None
    for _ in range(max_iter):
        alpha = compute_alpha_vectorized(B, m, q, mu0, Sigma0, bandwidth)

        optimizer.zero_grad()
        loss = compute_loss_vectorized(B, alpha, m, q, mu0, Sigma0, bandwidth)
        loss.backward()
        optimizer.step()

    final_B_error = torch.norm(B.detach() - A_true).item()
    final_alpha_error = torch.norm(alpha.detach() - b_true).item()
    return final_B_error, final_alpha_error


def run_rate_experiment_fast(n_values, bandwidth_values, num_repetitions):
    results = {}

    for h in bandwidth_values:
        results[h] = {"B": [], "alpha": []}

        print(f"\n=== Bandwidth h = {h} ===")
        for n in n_values:
            B_errors = []
            alpha_errors = []

            for _ in range(num_repetitions):
                B_err, alpha_err = run_single_experiment_fast(n=n, bandwidth=h)
                B_errors.append(B_err)
                alpha_errors.append(alpha_err)

            avg_B = float(np.mean(B_errors))
            avg_alpha = float(np.mean(alpha_errors))

            results[h]["B"].append(avg_B)
            results[h]["alpha"].append(avg_alpha)

            print(
                f"n = {n:4d} | "
                f"avg ||B_true - B|| = {avg_B:.6e} | "
                f"avg ||b_true - alpha|| = {avg_alpha:.6e}"
            )

    return results


def estimate_loglog_slope(n_values, errors):
    log_n = np.log(np.asarray(n_values, dtype=float))
    log_e = np.log(np.asarray(errors, dtype=float))
    slope, intercept = np.polyfit(log_n, log_e, 1)
    return slope, intercept


def plot_results_pretty(n_values, results, bandwidth_values):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # B panel
    ax = axes[0]
    for h in bandwidth_values:
        errs = results[h]["B"]
        slope, _ = estimate_loglog_slope(n_values, errs)
        ax.plot(
            n_values,
            errs,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=fr"$h={h}$ (slope {slope:.2f})"
        )

    # ref0 = results[bandwidth_values[0]]["B"][0] * math.sqrt(n_values[0])
    # ref_curve = [ref0 / math.sqrt(n) for n in n_values]
    # ax.plot(n_values, ref_curve, "--", linewidth=2, label=r"reference $n^{-1/2}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"Average final error in $B$")
    ax.set_title(r"Rate experiment for $B$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # alpha panel
    ax = axes[1]
    for h in bandwidth_values:
        errs = results[h]["alpha"]
        slope, _ = estimate_loglog_slope(n_values, errs)
        ax.plot(
            n_values,
            errs,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=fr"$h={h}$ (slope {slope:.2f})"
        )

    # ref0 = results[bandwidth_values[0]]["alpha"][0] * math.sqrt(n_values[0])
    # ref_curve = [ref0 / math.sqrt(n) for n in n_values]
    # ax.plot(n_values, ref_curve, "--", linewidth=2, label=r"reference $n^{-1/2}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"Average final error in $\alpha$")
    ax.set_title(r"Rate experiment for $\alpha$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle("Asymptotic rate experiment across bandwidths", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
    bandwidth_values = [0.25, 0.5, 1.0, 2.0]
    num_repetitions = 100

    results = run_rate_experiment_fast(
        n_values=n_values,
        bandwidth_values=bandwidth_values,
        num_repetitions=num_repetitions
    )

    plot_results_pretty(
        n_values=n_values,
        results=results,
        bandwidth_values=bandwidth_values
    )