import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility
# ============================================================
SEED = 123
rng = np.random.default_rng(SEED)

# ============================================================
# Settings
# ============================================================
MC_REPS = 200
N_GRID = np.array([100, 200, 400, 800, 1600, 3200, 6400])
EPS = 1e-12

m0 = 0.0
sigma0 = 1.0

# ============================================================
# Bandwidth: h = c * n^{-1/4}
# ============================================================
BANDWIDTH_CONST = 1.0

def bandwidth(n, c=BANDWIDTH_CONST):
    return c * n ** (-1/4)

# ============================================================
# Kernel
# ============================================================
def epanechnikov_kernel(u):
    u = np.asarray(u)
    out = 0.75 * (1.0 - u**2)
    out[np.abs(u) > 1.0] = 0.0
    return out

# ============================================================
# 1D Gaussian W2 distance
# ============================================================
def w2_1d_gaussian(m, sigma, m_ref=m0, sigma_ref=sigma0):
    return np.sqrt((m - m_ref)**2 + (sigma - sigma_ref)**2)

# ============================================================
# Fixed design distribution
# ============================================================
def sample_design(n, rng):
    m = rng.uniform(-1.5, 1.5, size=n)
    sigma = rng.uniform(0.5, 1.5, size=n)
    return m, sigma

# ============================================================
# True signal fields
# ============================================================
def alpha_true(m, sigma):
    dm = m - m0
    ds = sigma - sigma0
    return 1.0 + 0.25 * dm - 0.20 * ds + 0.30 * dm**2

def b_true(m, sigma):
    dm = m - m0
    ds = sigma - sigma0
    return 1.5 + 0.15 * dm + 0.10 * ds + 0.25 * ds**2

alpha0 = alpha_true(m0, sigma0)
b0 = b_true(m0, sigma0)

# ============================================================
# Noise
# ============================================================
ETA_STD = 0.20
B_EPS_STD = 0.10

def sample_noise(n, rng):
    eta_eps = rng.normal(0.0, ETA_STD, size=n)
    b_eps = 1.0 + rng.normal(0.0, B_EPS_STD, size=n)
    return eta_eps, b_eps

# ============================================================
# Data generation
# ============================================================
def generate_scalar_gaussian_data(n, rng):
    m, sigma = sample_design(n, rng)

    alpha_i = alpha_true(m, sigma)
    b_i = b_true(m, sigma)

    eta_eps, b_eps = sample_noise(n, rng)

    q = eta_eps + b_eps * (alpha_i + b_i * m)
    gamma = b_eps * b_i * sigma

    return m, sigma, q, gamma

# ============================================================
# Kernel weights
# ============================================================
def kernel_weights(m, sigma, h):
    d = w2_1d_gaussian(m, sigma, m0, sigma0)
    return epanechnikov_kernel(d / (h + EPS))

# ============================================================
# Closed-form estimators using factored sums
# ============================================================
def estimate_alpha_b(m, sigma, q, gamma, K):
    S0 = np.sum(K)
    Sq = np.sum(q * K)
    Sm = np.sum(m * K)
    Ss = np.sum((m**2 + sigma**2) * K)
    Smq = np.sum(m * q * K)
    Sgs = np.sum(gamma * sigma * K)

    den = S0 * Ss - Sm**2

    alpha_num = Sq * Ss - Sm * (Smq + Sgs)
    b_num = S0 * (Smq + Sgs) - Sm * Sq

    alpha_hat = alpha_num / (den + EPS)
    b_hat = b_num / (den + EPS)

    return alpha_hat, b_hat

# ============================================================
# Containers
# ============================================================
alpha_errors = np.full((MC_REPS, len(N_GRID)), np.nan)
b_errors = np.full((MC_REPS, len(N_GRID)), np.nan)

# ============================================================
# Monte Carlo loop
# ============================================================
for n_idx, n in enumerate(N_GRID):
    h = bandwidth(n)
    print(f"Running n = {n}, h = {h:.4f}")

    for mc in range(MC_REPS):
        m, sigma, q, gamma = generate_scalar_gaussian_data(n, rng)
        K = kernel_weights(m, sigma, h)

        if np.sum(K) < 5:
            continue

        alpha_hat, b_hat = estimate_alpha_b(m, sigma, q, gamma, K)

        alpha_errors[mc, n_idx] = abs(alpha_hat - alpha0)
        b_errors[mc, n_idx] = abs(b_hat - b0)

# ============================================================
# Average errors over Monte Carlo repetitions
# ============================================================
alpha_mean_err = np.nanmean(alpha_errors, axis=0)
b_mean_err = np.nanmean(b_errors, axis=0)

print("\nalpha mean errors:")
print(alpha_mean_err)

print("\nb mean errors:")
print(b_mean_err)

# ============================================================
# Estimate slopes via log-log regression
# ============================================================
log_n = np.log(N_GRID)

log_alpha_err = np.log(alpha_mean_err)
log_b_err = np.log(b_mean_err)

# Fit: log(error) = a + slope * log(n)
alpha_slope, alpha_intercept = np.polyfit(log_n, log_alpha_err, 1)
b_slope, b_intercept = np.polyfit(log_n, log_b_err, 1)

print("\nEstimated convergence rates (log-log slopes):")
print(f"alpha slope ≈ {alpha_slope:.4f}")
print(f"b slope     ≈ {b_slope:.4f}")

# ============================================================
# Plot: error vs n
# ============================================================
plt.figure(figsize=(8, 5))
plt.loglog(N_GRID, alpha_mean_err, marker='o', label=r'$|\hat{\alpha}_n - \alpha_0|$')
plt.loglog(N_GRID, b_mean_err, marker='s', label=r'$|\hat{b}_n - b_0|$')

plt.xlabel("n")
plt.ylabel("Mean absolute error")
plt.title(r"Estimator error vs $n$ with $h_n = n^{-1/4}$")

# ============================================================
# Reference slope line: n^{-1/4}
# ============================================================
ref_n = N_GRID.astype(float)

# anchor the line at the first alpha error (you can change this)
anchor = b_mean_err[0]

ref_line = anchor * (ref_n / ref_n[0])**(-1/4)

plt.loglog(
    ref_n,
    ref_line,
    'k--',
    linewidth=2,
    label=r'Ref slope $n^{-1/4}$'
)

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()