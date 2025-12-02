# src/data/metadata.py

import torch
from pathlib import Path

def load_metadata(dim, base_dir="data/generated/metadata", n=1000, k=100):
    path = Path(base_dir) / f"gmm_pairs_n{n}_k{k}_d{dim}.pt"
    return torch.load(path)

def load_test_pair(k, dim, base_dir="data/generated/test-pairs"):
    path = Path(base_dir) / f"gmm_pairs_n1_k100_d{dim}.pt"
    data = torch.load(path)
    return data['mu'][0][:k], data['nu'][0][:k]
