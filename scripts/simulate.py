# scripts/simulate.py

import torch
from pathlib import Path
from src.data_utils.generator import generate_gaussian_pairs, generate_2d_gmm_pairs,  generate_5d_gmm_pairs
def main():
    # === Config ===
    n_distributions = 1
    n_samples = 1000
    dim = 5
    seed = 42
    n_components = 3
    output_dir = Path("data/generated/test-pairs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Generate data ===
    mu_tensor, nu_tensor = generate_5d_gmm_pairs(
        n_distributions=n_distributions,
        n_samples=n_samples,
        dim=dim,
        n_components=n_components,
        seed=seed
    )

    print(f"Generated {n_distributions} distribution pairs.")
    print(f"Shape of mu_tensor: {mu_tensor.shape}")
    print(f"Shape of nu_tensor: {nu_tensor.shape}")

    # === Save to disk ===
    data_path = output_dir / f"gmm_pairs_n{n_distributions}_k{n_samples}_d{dim}.pt"
    torch.save({'mu': mu_tensor, 'nu': nu_tensor}, data_path)

    print(f"Saved dataset to: {data_path}")


if __name__ == "__main__":
    main()
