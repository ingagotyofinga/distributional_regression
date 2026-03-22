import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility
# ============================================================
SEED = 123
rng = np.random.default_rng(SEED)

# ============================================================
# Settings for CLT experiment
# ============================================================
MC_REPS = 4000
n = 20000                 # large n for CLT
EPS = 1e-12

m0 = 0.0
sigma0 = 1.0

# ============================================================
# Bandwidth: h = c * n^{-1/4}
# ============================================================
BANDWIDTH_CONST = 1.0

def bandwidth(n, c=BANDWIDTH_CONST):
    return c * n ** (-1/4)

h = bandwidth(n)

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
# Fixed/random design distribution
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

    return alpha_hat, b_hat, S0

# ============================================================
# Theoretical variance constants
# ============================================================
# design density at (m0, sigma0) for Uniform([-1.5,1.5] x [0.5,1.5])
f0 = 1.0 / (3.0 * 1.0)   # = 1/3

# radial 2D Epanechnikov kernel integrals
int_K = 3.0 * np.pi / 8.0
int_K2 = 3.0 * np.pi / 16.0

kernel_constant = int_K2 / (f0 * int_K**2)   # simplifies to 4/pi

V_alpha_theory = (ETA_STD**2 + alpha0**2 * B_EPS_STD**2) * kernel_constant
V_b_theory = (b0**2 * B_EPS_STD**2) * kernel_constant

print(f"n = {n}")
print(f"h = {h:.6f}")
print(f"sqrt(n h^2) = {np.sqrt(n * h**2):.6f}")
print(f"Theoretical V_alpha = {V_alpha_theory:.6f}")
print(f"Theoretical V_b     = {V_b_theory:.6f}")

# ============================================================
# Monte Carlo for CLT
# ============================================================
alpha_hats = []
b_hats = []
Z_alpha = []
Z_b = []
support_sizes = []

norm_factor = np.sqrt(n * h**2)

for mc in range(MC_REPS):
    m, sigma, q, gamma = generate_scalar_gaussian_data(n, rng)
    K = kernel_weights(m, sigma, h)

    alpha_hat, b_hat, S0 = estimate_alpha_b(m, sigma, q, gamma, K)

    # store results
    alpha_hats.append(alpha_hat)
    b_hats.append(b_hat)
    support_sizes.append(np.sum(K > 0))

    Z_alpha.append(norm_factor * (alpha_hat - alpha0))
    Z_b.append(norm_factor * (b_hat - b0))

alpha_hats = np.array(alpha_hats)
b_hats = np.array(b_hats)
Z_alpha = np.array(Z_alpha)
Z_b = np.array(Z_b)
support_sizes = np.array(support_sizes)

# ============================================================
# Empirical summaries
# ============================================================
print("\nEmpirical estimator summaries:")
print(f"mean(alpha_hat) = {np.mean(alpha_hats):.6f}")
print(f"mean(b_hat)     = {np.mean(b_hats):.6f}")
print(f"mean support size = {np.mean(support_sizes):.2f}")

print("\nCLT summaries for normalized statistics:")
print(f"mean(Z_alpha) = {np.mean(Z_alpha):.6f}")
print(f"var(Z_alpha)  = {np.var(Z_alpha, ddof=1):.6f}")
print(f"theory        = {V_alpha_theory:.6f}")

print()

print(f"mean(Z_b)     = {np.mean(Z_b):.6f}")
print(f"var(Z_b)      = {np.var(Z_b, ddof=1):.6f}")
print(f"theory        = {V_b_theory:.6f}")

# ============================================================
# Normal density helper
# ============================================================
def normal_pdf(x, mean, var):
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-(x - mean)**2 / (2.0 * var))

# ============================================================
# Plot histograms with theoretical Gaussian overlays
# ============================================================
# alpha
x_alpha = np.linspace(
    np.min(Z_alpha) - 0.5,
    np.max(Z_alpha) + 0.5,
    400
)

plt.figure(figsize=(8, 5))
plt.hist(Z_alpha, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.plot(
    x_alpha,
    normal_pdf(x_alpha, mean=0.0, var=V_alpha_theory),
    linewidth=2,
    label=rf'Theory $N(0, {V_alpha_theory:.4f})$'
)
plt.axvline(np.mean(Z_alpha), linestyle='--', linewidth=2, label=f'Empirical mean = {np.mean(Z_alpha):.4f}')
plt.title(rf'CLT check for $\sqrt{{n h^2}}(\hat{{\alpha}}-\alpha_0)$, $n={n}$')
plt.xlabel(r'$\sqrt{n h^2}(\hat{\alpha}-\alpha_0)$')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# b
x_b = np.linspace(
    np.min(Z_b) - 0.5,
    np.max(Z_b) + 0.5,
    400
)

plt.figure(figsize=(8, 5))
plt.hist(Z_b, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.plot(
    x_b,
    normal_pdf(x_b, mean=0.0, var=V_b_theory),
    linewidth=2,
    label=rf'Theory $N(0, {V_b_theory:.4f})$'
)
plt.axvline(np.mean(Z_b), linestyle='--', linewidth=2, label=f'Empirical mean = {np.mean(Z_b):.4f}')
plt.title(rf'CLT check for $\sqrt{{n h^2}}(\hat{{b}}-b_0)$, $n={n}$')
plt.xlabel(r'$\sqrt{n h^2}(\hat{b}-b_0)$')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()