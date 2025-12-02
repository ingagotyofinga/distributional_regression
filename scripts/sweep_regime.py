# scripts/sweep_regime.py
import os, sys

def sweep_regime(regime_id, dim, n, k, output_dir, mode="nn", seed=42, num_runs=10, start_subset_seed=0):
    for ss in range(start_subset_seed, start_subset_seed + num_runs):
        log_dir = f"{output_dir}/gmm_{regime_id}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/log_sub{ss}.txt"
        if os.path.exists(log_file):
            print(f"✅ Skipping {regime_id} subset_seed {ss} (already run)")
            continue
        print(f"🚀 Launching: {regime_id} subset_seed {ss}")
        cmd = (
            "PYTHONPATH=. python scripts/run_regime.py "
            f"--regime_id {regime_id} --dim {dim} --n {n} --k {k} "
            f"--mode {mode} --seed {seed} --subset_seed {ss} "
            f"--output_dir {output_dir} "
        )
        os.system(f"{cmd} > {log_file} 2>&1")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python scripts/sweep_regime.py <regime_id> <dim> <n> <k> [mode] [seed] [num_runs]")
        sys.exit(1)
    regime_id = sys.argv[1]
    dim = int(sys.argv[2]); n = int(sys.argv[3]); k = int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 else "nn"
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42
    num_runs = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    output_dir = f"results/d{dim}"
    sweep_regime(regime_id, dim, n, k, output_dir, mode=mode, seed=seed, num_runs=num_runs)
