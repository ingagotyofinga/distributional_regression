# scripts/run_regime.py
import argparse
from pathlib import Path
import subprocess

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regime_id", type=str, required=True)          # e.g. "A"
    p.add_argument("--dim", type=int, required=True)                # d used by training
    p.add_argument("--n", type=int, required=True)                  # subsample #dists
    p.add_argument("--k", type=int, required=True)                  # subsample #points/dist
    p.add_argument("--mode", type=str, required=True, choices=["nn", "affine"])
    p.add_argument("--seed", type=int, required=True)               # training seed
    p.add_argument("--subset_seed", type=int, required=True)        # subsampling seed
    p.add_argument("--output_dir", type=str, required=True)
    # NEW: master dataset controls (so we don't derive path from (n,k))
    p.add_argument("--master_n", type=int, default=10000)
    p.add_argument("--master_k", type=int, default=10000)
    p.add_argument("--meta_dir", type=str, default="data/generated/metadata")
    return p.parse_args()

def main():
    args = parse_args()

    # Always point to the MASTER file; train_gmm.py does the subsampling to (n,k)
    master_path = Path(args.meta_dir) / f"gmm_pairs_n{args.master_n}_k{args.master_k}_d{args.dim}.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-u", "scripts/train_gmm.py",
        "--regime_id", args.regime_id,
        "--dim", str(args.dim),
        "--n", str(args.n),
        "--k", str(args.k),
        "--seed", str(args.seed),
        "--subset_seed", str(args.subset_seed),
        "--input_file", str(master_path),          # <— master file here
        "--output_dir", args.output_dir,
        "--mode", args.mode,
    ]

    log_dir = Path(args.output_dir) / f"gmm_{args.regime_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_sub{args.subset_seed}.txt"

    print("[run_regime]", " ".join(cmd), flush=True)
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"[run_regime] return code: {result.returncode}")
    print(f"[run_regime] log: {log_file}")

if __name__ == "__main__":
    main()
