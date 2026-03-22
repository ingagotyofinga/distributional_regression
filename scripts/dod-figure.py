import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)

# -----------------------------
# Simulate paired distributions
# -----------------------------
n_pairs = 3
n_samples = 1000

# input distribution parameters
m_vals = np.array([-1.2, 0.0, 1.0])
s_vals = np.array([0.45, 0.7, 0.35])

# simple regression relationship on distribution parameters
def response_params(m, s):
    q = 1.2 + 1.4 * m
    t = 0.35 + 0.8 * s
    return q, t

mu_samples = []
nu_samples = []

for m, s in zip(m_vals, s_vals):
    x = rng.normal(m, s, size=n_samples)
    q, t = response_params(m, s)
    y = rng.normal(q, t, size=n_samples)
    mu_samples.append(x)
    nu_samples.append(y)

# "new" input and corresponding prediction target
m0, s0 = 0.5, 0.55
mu0 = rng.normal(m0, s0, size=n_samples)
q0, t0 = response_params(m0, s0)
nu0_hat = rng.normal(q0, t0, size=n_samples)

# common x-range
all_data = np.concatenate(mu_samples + nu_samples + [mu0, nu0_hat])
xmin, xmax = all_data.min() - 0.5, all_data.max() + 0.5

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(
    2, 4, figsize=(12, 5), sharex=True, sharey=True,
    gridspec_kw={"wspace": 0.28, "hspace": 0.45}
)

input_color = "#4C78A8"
output_color = "#F58518"
highlight_color = "#54A24B"

bins = np.linspace(xmin, xmax, 18)

# training pairs
for j in range(3):
    ax_top = axes[0, j]
    ax_bot = axes[1, j]

    ax_top.hist(mu_samples[j], bins=bins, density=True,
                histtype="stepfilled", alpha=0.30, color=input_color,
                edgecolor=input_color, linewidth=2)
    ax_top.set_title(rf"$\mu_{{{j+1}}}$", fontsize=16, color=input_color, pad=6)

    ax_bot.hist(nu_samples[j], bins=bins, density=True,
                histtype="stepfilled", alpha=0.30, color=output_color,
                edgecolor=output_color, linewidth=2)
    ax_bot.set_title(rf"$\nu_{{{j+1}}}$", fontsize=16, color=output_color, pad=6)

# new input / predicted output
axes[0, 3].hist(mu0, bins=bins, density=True,
                histtype="stepfilled", alpha=0.30, color=highlight_color,
                edgecolor=highlight_color, linewidth=2)
axes[0, 3].set_title(r"$\mu_0$", fontsize=16, color=highlight_color, pad=6)

axes[1, 3].hist(nu0_hat, bins=bins, density=True,
                histtype="stepfilled", alpha=0.25, color=highlight_color,
                edgecolor=highlight_color, linewidth=2, linestyle="--")
axes[1, 3].set_title(r"$\hat{\nu}_0$", fontsize=16, color=highlight_color, pad=6)

# clean up axes
for ax in axes.ravel():
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.show()