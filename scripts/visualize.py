import torch
from pathlib import Path
from src.utils.plot import plot_distribution_pair

def main():
    # === Load dataset ===
    data_path = Path("data/generated/metadata/gmm_pairs_n4_k100_d2.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = torch.load(data_path)
    mu_tensor = data['mu']
    nu_tensor = data['nu']

    print(f"Loaded {mu_tensor.shape[0]} pairs of distributions.")

    # === Plot a few examples
    for idx in range(3):
        plot_distribution_pair(mu_tensor[idx], nu_tensor[idx], index=idx)

if __name__ == "__main__":
    main()
