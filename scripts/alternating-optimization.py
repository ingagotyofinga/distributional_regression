import torch
import matplotlib.pyplot as plt

# Define constants and initializations
n = 100  # sample size
dim = 2  # dimension of the distributions
A = 2*torch.eye(dim)
b = 5
E = 1e-2*torch.randn(dim, dim)
err = [E @ E.T for _ in range(n)]
m = torch.zeros(n, dim, 1) + torch.randn(n,dim,1)  # source means (n x dim x 1)
q = A @ m + b  # target means (n x dim x 1)
Sigma = [torch.eye(dim) for _ in range(n)]  # source covariance matrices (dim x dim)
Gamma = [A @ sigma @ A.T + e for sigma, e in zip(Sigma, err)] # target covariance matrices (dim x dim)
mu0 = torch.randn(dim, 1)  # reference mean
Sigma0 = torch.eye(dim)  # reference covariance matrix

# Compute expected B using the formula: Σ^(-1/2) * (Σ^(1/2) Γ Σ^(1/2)) Σ^(-1/2)
def compute_expected_B(Sigma, Gamma):
    Sigma_sqrt = torch.linalg.cholesky(Sigma)
    Sigma_inv_sqrt = torch.linalg.inv(Sigma_sqrt)
    middle_term = torch.linalg.cholesky(Sigma_sqrt @ Gamma @ Sigma_sqrt.T)
    return Sigma_inv_sqrt @ middle_term @ Sigma_inv_sqrt.T

# Compute expected alpha using the formula: alpha = q.mean - B * m.mean
def compute_expected_alpha(B, m, q):
    m_mean = m.mean(dim=0)
    q_mean = q.mean(dim=0)
    return q_mean - B @ m_mean

# Wasserstein distance between Gaussians
def wasserstein_distance_gaussian(mu1, Sigma1, mu2, Sigma2):
    mean_dist = torch.norm(mu1 - mu2) ** 2
    Sigma1_sqrt = torch.linalg.cholesky(Sigma1)  # Square root of Sigma1
    inner_term = Sigma1_sqrt @ Sigma2 @ Sigma1_sqrt.T
    trace_term = torch.trace(Sigma1 + Sigma2 - 2 * torch.linalg.cholesky(inner_term))
    return mean_dist + trace_term


# Gaussian Wasserstein-based kernel
def wasserstein_kernel(mu0, Sigma0, mui, Sigmai, bandwidth=1.0):
    w2_squared = wasserstein_distance_gaussian(mu0, Sigma0, mui, Sigmai)
    return torch.exp(-w2_squared / (2 * bandwidth ** 2))


# Compute Y_i = Σ_i^(1/2) * Γ_i * Σ_i^(1/2)
def compute_Y(Sigma_i, Gamma_i):
    Sigma_i_sqrt = torch.linalg.cholesky(Sigma_i)
    return Sigma_i_sqrt @ Gamma_i @ Sigma_i_sqrt.T


# Compute alpha based on current estimate of B
def compute_alpha(B, m, q, mu0, Sigma0, Sigma, bandwidth=1.0, epsilon=0):
    numerator = sum(
        (-B @ m[i] + q[i]) * wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)
        for i in range(n)
    )
    denominator = sum(wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth) for i in range(n))

    # print("Numerator:", numerator)
    # print("Denominator:", denominator)

    return numerator / (denominator + epsilon)


# Compute B based on current estimate of alpha
def compute_B(alpha, m, q, Sigma, Gamma, mu0, Sigma0, bandwidth=1.0, epsilon=0):
    gradient = 0
    for i in range(n):
        term1 = 2 * (alpha + B @ m[i] - q[i]) @ m[i].T
        term2 = 2 * B @ Sigma[i]
        term3 = B @ compute_Y(Sigma[i], Gamma[i]) @ B.T + epsilon * torch.eye(B.shape[0])
        term3 = torch.linalg.cholesky(term3)
        term3 = torch.linalg.inv(term3)
        term4 = term3 * (compute_Y(Sigma[i], Gamma[i]) @ B.T + B @ compute_Y(Sigma[i], Gamma[i]).T)

        kernel_value = wasserstein_kernel(mu0, Sigma0, m[i], Sigma[i], bandwidth)

        # print(f"Iteration {i}, Term1:", term1)
        # print(f"Iteration {i}, Term2:", term2)
        # print(f"Iteration {i}, Term3:", term3)
        # print(f"Iteration {i}, Term4:", term4)
        # print(f"Iteration {i}, Kernel Value:", kernel_value)

        gradient += (term1 + term2 - term4) * kernel_value
    gradient_norm = torch.norm(gradient)
    # print("Gradient:", gradient)
    return B - learning_rate * (gradient/gradient_norm)


# Initialize B and set optimization parameters
B = torch.eye(dim) + 0.01 * torch.randn(dim, dim)  # initialize B as a random dim x dim matrix
tolerance = 1e-8
max_iter = 100
learning_rate = 1e-1

# Lists to store alpha and B values for each iteration
alpha_values = []
B_norms = []
difference_norms = []
B_news = []

# Alternating optimization loop
for iteration in range(max_iter):
    # Step 1: Compute alpha with current B
    alpha = compute_alpha(B, m, q, mu0, Sigma0, Sigma)
    alpha_values.append(alpha.clone().detach().mean().item())  # Mean value for plot
    # print(f"Iteration {iteration}, Alpha:", alpha)

    # Step 2: Update B based on the computed alpha
    B_new = compute_B(alpha, m, q, Sigma, Gamma, mu0, Sigma0)
    B_news.append(B_new)
    B_norms.append(torch.norm(B_new).item())  # Norm of B for plot
    # print(f"Iteration {iteration}, B_new:", B_new)
    difference_norms.append(torch.norm(B_new - B).item())

    # Check for convergence
    if difference_norms[-1] < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break
    B = B_new

# Calculate expected B and alpha based on the final optimized values
B_expected = compute_expected_B(Sigma0, Gamma[0])  # Using the reference covariance matrix
alpha_expected = compute_expected_alpha(B, m, q)

print("m mean:", m.mean(dim=0))
print("q mean:", q.mean(dim=0))
print("Sigma:", Sigma[0])
print("Gamma:", Gamma[0])


# Evaluate results
print("\nTesting Results:")
print("Expected alpha:", alpha_expected)
print("Expected B:", B_expected)
print("Estimated alpha:", alpha)
print("Estimated B:", B)

# Calculate errors
alpha_error = torch.norm(alpha - alpha_expected)
B_error = torch.norm(B - B_expected)

print(f"\nAlpha error: {alpha_error.item():.4f}")
print(f"B error: {B_error.item():.4f}")
# Plot alpha and B over iterations
plt.figure(figsize=(15, 6))

# Plot alpha
plt.subplot(1, 3, 1)
plt.plot(alpha_values, label="Alpha")
plt.xlabel("Iteration")
plt.ylabel("Alpha Value (Mean)")
plt.title("Alpha over Iterations")
plt.legend()

# Plot B norm
plt.subplot(1, 3, 2)
plt.plot(B_norms, label="Norm of B")
plt.xlabel("Iteration")
plt.ylabel("B Norm")
plt.title("Norm of B over Iterations")
plt.legend()

# Plot the norm of B_expected - B over iterations
norm_diffs = [torch.norm(B_expected - B_new).item() for B_new in B_news]
plt.subplot(1, 3, 3)
plt.plot(norm_diffs, label="||B_expected - B||")
plt.xlabel("Iteration")
plt.ylabel("Difference Norm")
plt.title("Norm of B_expected - B over Iterations")
plt.title("Norm of B_expected - B over Iterations")
plt.legend()

plt.tight_layout()
plt.show()