import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import numpy as np
from geomloss import SamplesLoss
from pathlib import Path
from src.estimator import DistributionToAffineMapNet
from src.data_utils.pair_dataset import DistributionPairDataset
from statsmodels.nonparametric.smoothers_lowess import lowess
from src.utils.covering_references import select_covering_references

# --- CONFIG ---
PATH_TO_DATA = "data/generated/pairs_n100_k1000_d2.pt"
MODEL_DIR = Path("figures/phase7 - normalized loss curves/blur0.4_lr0.0001")
REFERENCE_JSON = MODEL_DIR / "reference_ids.json"
OUTPUT_DIR = Path("figures/distance_effects")
BATCH_SIZE = 1
BLUR = 0.1
N_BINS = 6

# --- SETUP ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- LOAD DATA ---
dataset = DistributionPairDataset(PATH_TO_DATA)
train_dataset, val_dataset = dataset.train_val_split(seed=42)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_indices = train_dataset.indices
mu_train = dataset.mu[train_indices]

# Recompute reference indices
reference_local_ids, _, _ = select_covering_references(mu_train, h=0.4, C=0.05)
reference_ids = [train_indices[idx] for idx in reference_local_ids]

# Save them back to JSON
with open(REFERENCE_JSON, "w") as f:
    json.dump(reference_ids, f)

print("Recovered and saved reference_ids.json")

# --- DEFINE LOSSES ---
sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=BLUR)

def w2_squared(x, y):
    return sinkhorn_loss(x, y)

# --- EVALUATE ALL REFERENCE MODELS ---
all_distances = []
all_errors = []

for j, reference_idx in enumerate(reference_ids):
    print(f"Evaluating reference {j} (global index {reference_idx})")
    mu_0 = dataset.mu[reference_idx]

    model_path = MODEL_DIR / f"mu0_{j}" / "best_model.pt"
    model = DistributionToAffineMapNet(input_dim=2, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    distances = []
    errors = []

    for batch in val_loader:
        mu_i, nu_i = batch
        mu_i, nu_i = mu_i.squeeze(0), nu_i.squeeze(0)

        x = mu_i
        with torch.no_grad():
            alpha, B = model(mu_i)
            T_mu_i = (B @ x.T + alpha[:, None]).T

        d = sinkhorn_loss(mu_i, mu_0)
        e = sinkhorn_loss(T_mu_i, nu_i)

        distances.append(d.item())
        errors.append(e.item())

    all_distances.extend(distances)
    all_errors.extend(errors)

# --- SAVE RAW RESULTS ---
np.savez(OUTPUT_DIR / "distance_effects.npz", distances=all_distances, errors=all_errors)

# --- BINNED SUMMARY STATS ---
bins = np.linspace(min(all_distances), max(all_distances), N_BINS + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
mean_errors = []
std_errors = []

for i in range(N_BINS):
    mask = (np.array(all_distances) >= bins[i]) & (np.array(all_distances) < bins[i+1])
    bin_errors = np.array(all_errors)[mask]
    if len(bin_errors) > 0:
        mean_errors.append(np.mean(bin_errors))
        std_errors.append(np.std(bin_errors))
    else:
        mean_errors.append(np.nan)
        std_errors.append(np.nan)

# --- PLOT ---
plt.figure(figsize=(8, 6))
plt.scatter(all_distances, all_errors, alpha=0.6, label="Samples")

# Fit and plot LOWESS curve
smoothed = lowess(all_errors, all_distances, frac=0.4)
plt.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=2, label="LOWESS")

plt.xlabel("W2 Distance to Reference ($\\mu_0$)")
plt.ylabel("Prediction Error ($W_2^2(T_{\\mu_0}\\#\\mu_i, \\nu_i)$)")
plt.title("Prediction Error vs Distance from Reference (All References)")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distance_effects.png")
plt.show()
