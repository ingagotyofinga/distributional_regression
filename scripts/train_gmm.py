# scripts/train.py

import argparse
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.data import Subset
matplotlib.use('Agg')
import sys
from src.data_utils.pair_dataset import DistributionPairDataset
from src.local_fit import fit_local_map
from src.utils.plot import visualize_pushforward, loss_curves
from src.utils.log_experiment import log_experiment
from src.loss import sinkhorn_cost
from src.utils.kernel import knn_bandwidth
from src.data_utils.metadata import load_test_pair

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--n", type=int, required=True, help="Subsample this many training pairs after loading")
    p.add_argument("--k", type=int, required=False,
                   help="Samples per distribution; used by load_test_pair(k, dim).")
    p.add_argument("--seed", type=int, required=True, help="Random seed for model.")
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["affine", "nn"])
    p.add_argument("--bw", type=float, default=0.085, help="Kernel bandwidth for local loss.")
    p.add_argument("--subset_seed", type=int, default=0,
                   help="Seed for the random subsample of pairs.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--regime_id", type=str, required=True,
                   help="Letter A–F matching sweep_all_regimes; used in paths & logs.")
    return p.parse_args()

def set_seed(seed: int):
    # defines seed for randomness coming from torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

def scatter_pushforward(mu0: torch.Tensor, yhat: torch.Tensor, nu0: torch.Tensor, out_path: Path):
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(mu0[:, 1], mu0[:, 0], s=2, label="μ₀")
    plt.scatter(yhat[:, 1], yhat[:, 0], s=2, label="T#μ₀ (pred)")
    plt.scatter(nu0[:, 1], nu0[:, 0], s=2, label="ν₀ (target)", alpha=0.6)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()

    print("🟢 Script started", flush=True)
    sys.stdout.flush()
    set_seed(args.seed)

    # === Paths and Config ===
    base_output_dir = Path(args.output_dir) / f"gmm_{args.regime_id}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    experiment_log_file = base_output_dir / "experiment_log.csv"
    run_name = f"mu0_s{args.seed}_sub{args.subset_seed}_n{args.n}_k{args.k}"
    phase_dir = base_output_dir / run_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    n_epochs = int(args.epochs)
    patience = int(args.patience)

    print("📦 Loading data...", flush=True)
    dataset = DistributionPairDataset(
        path=args.input_file,  # <- master file path
        data_kind="gmm",
        n_sub=args.n,  # <- subsample to n distributions
        k_sub=args.k,  # <- subsample to k points per distribution
        subset_seed=args.subset_seed,
        verbose=True,
    )

    # === Confirm shapes ===
    if hasattr(dataset, "shape") and dataset.shape is not None:
        N_used, K_used, d_used = dataset.shape
        print(f"🧪 Using dataset shape: (N={N_used}, K={K_used}, d={d_used}) "
              f"from master='{args.input_file}' with subset_seed={args.subset_seed}")
        # keep args in sync with reality, in case master is smaller than requested
        args.n = N_used
        args.k = K_used
    else:
        # weighted path or ragged (not your case for GMM)
        print("🧪 Using weighted/ragged dataset")

    # === Define Test Set ===
    src_test, tgt_test = load_test_pair(args.k, args.dim)
    val_list = [(src_test, tgt_test)]

    # === Compute Bandwidth as function of k-NN ===
    print("🔍 Selecting kNN bandwidth...", flush=True)

    n_train = int(args.n)
    k = int(np.clip(np.ceil(0.20 * n_train), 1, n_train))  # 20% neighbors
    src_train = [dataset[j][0] for j in range(n_train)]  # collect μ_i only

    bw = knn_bandwidth(src_test, src_train, k=k)  # returns length scale h
    print(f"🔍 kNN bandwidth: k={k}/{n_train} → bw={bw:.4f}", flush=True)

    # === Compute W2 of Identity Map for Loss Ratio ===
    id_base = float(sinkhorn_cost(src_test, tgt_test))
    print(f"[calib]  val_id_base={id_base:.6f}")

    print("🧠 Calling fit_local_map...", flush=True)
    start_time = time.time()

    result = fit_local_map(
        src_test,
        dataset,          # train Dataset
        val_list,         # 1-item val (weighted or sampled)
        input_dim=args.dim,
        hidden_dim=128,
        bw=bw,
        lr=1e-3,
        n_epochs=n_epochs,
        batch_size=batch_size,
        patience=patience,
        phase_dir=phase_dir,
        verbose=True,
        mode=args.mode,
        base_val=id_base,
        device=DEVICE,
    )

    val_losses = np.array(result["val_losses"])
    best_idx = int(val_losses.argmin())
    best_val = float(val_losses[best_idx])
    best_ratio = best_val/max(id_base,1e-12)
    print(f"🏅 Best epoch = {best_idx} with val_loss = {best_val:.6g} and val_loss_ratio = {best_ratio:.6g}")

    elapsed = time.time() - start_time
    print("✅ fit_local_map returned!", flush=True)
    print(f"⏱️ Training completed in {elapsed:.2f} seconds", flush=True)

    print("📊 Saving visualizations...", flush=True)

    if args.mode == "affine":
        for epoch, (alpha, B) in enumerate(zip(result["alpha_history"], result["B_history"])):
            if epoch % 1 == 0:
                yhat = alpha + src_test @ B.T
                scatter_pushforward(src_test, yhat, tgt_test, phase_dir / f"pushforward_epoch_{epoch}.png")
    else:
        y_hist = result.get("y_hat_history", [])
        if isinstance(y_hist, (list, tuple)) and len(y_hist) > 0:
            for epoch, y_hat in enumerate(y_hist):
                if epoch % 7 == 0:
                    scatter_pushforward(src_test, y_hat, tgt_test, phase_dir / f"pushforward_epoch_{epoch}.png")

    loss_curves(result["train_losses"], result["val_losses"], phase_dir)

    with open(phase_dir / "best_epoch.txt", "w") as f:
        f.write(f"{best_idx}\n")

    final_val_loss = result["val_losses"][-1]
    val_record = {
        "regime_id": args.regime_id,
        "d": args.dim,
        "n": args.n,
        "k": args.k,
        "bw": bw,
        "test_err_best": best_val,
        "id_base": id_base,
        "best_ratio": best_ratio,
        "train_time_sec": round(elapsed, 2),
        "map_seed": args.seed,
        "subset_seed": args.subset_seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_experiment(val_record, experiment_log_file)
    print(f"✅ Logged result: val_loss = {final_val_loss:.4f}\n", flush=True)

if __name__ == "__main__":
    main()
