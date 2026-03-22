# scripts/sweep_all_regimes.py
from scripts.sweep_regime import sweep_regime

# regimes = {
#     "A": (2, 100, 1000),
#     "B": (2, 100, 100),
#     "C": (2, 100, 5000),
#     "D": (2, 10, 1000),
#     "E": (2, 1000, 1000),
#     "F": (5, 100, 1000),
#     "G": (5, 100, 100),
#     "H": (5, 100, 10),
#     "I": (5, 10, 100)
# }

regimes = {
    "I": (5, 1000, 1000)
}

# regimes = {
#     "Meep": (2, 100, 100),
# }

MODE = "nn"   # or "affine"
SEED = 42     # fixed training seed
RUNS = 10     # number of subset_seed draws

for regime_id, (dim, n, k) in regimes.items():
    output_dir = f"results/d{dim}"
    print(f"\n=== Sweeping regime {regime_id} (d={dim}, n={n}, k={k}) ===")
    sweep_regime(regime_id, dim, n, k, output_dir, mode=MODE, seed=SEED, num_runs=RUNS)
