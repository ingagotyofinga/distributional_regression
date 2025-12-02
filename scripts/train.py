# scripts/train.py

import argparse
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib
from src.utils.plot import loss_curves_with_ratio
import numpy as np
from torch.utils.data import Subset


matplotlib.use('Agg')
import sys


from src.data_utils.pair_dataset import DistributionPairDataset
from src.local_fit import fit_local_map
from src.utils.plot import visualize_pushforward, loss_curves, visualize_pointcloud, pointcloud_to_image
from src.utils.log_experiment import log_experiment
from src.utils.kernel import knn_bandwidth
from src.loss import sinkhorn_cost
from src.data_utils.metadata import load_test_pair

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--n", type=int, required=False)
    parser.add_argument("--k", type=int, required=False)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--blur", type=str, required=False)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Subsample this many training pairs after loading")
    parser.add_argument("--subset_seed", type=int, default=0,
                        help="Seed for the random subsample of pairs")

    args = parser.parse_args()

    print("🟢 Script started", flush=True)
    sys.stdout.flush()

    torch.manual_seed(args.seed)

    # === Paths and Config ===
    base_output_dir = Path(args.output_dir) / args.regime
    base_output_dir.mkdir(parents=True, exist_ok=True)
    experiment_log_file = base_output_dir / "experiment_log.csv"

    batch_size = 1
    n_epochs = 200
    patience = 7

    print("📦 Loading data...", flush=True)

    if args.regime.lower() == "mnist":
        if args.dim != 2:
            raise ValueError("For MNIST experiments, you must set --dim=2")

        dataset = DistributionPairDataset(args.input_file)

        def _len(ds):
            return len(ds)

        def _take_subset(ds, m, seed):
            import torch
            g = torch.Generator().manual_seed(seed)
            n_total = len(ds)
            m = min(m, n_total)
            idx = torch.randperm(n_total, generator=g)[:m].tolist()
            sub = Subset(ds, idx)
            # keep meta accessible even when wrapped
            base = ds
            setattr(sub, "meta", getattr(base, "meta", {}))
            return sub, m, n_total

        if args.max_pairs is not None:
            dataset, used, total = _take_subset(dataset, args.max_pairs, args.subset_seed)
            print(f"🧪 Using {used}/{total} training pairs (subsampled)")
            # keep the 'n' label in your output path honest
            args.n = used if (args.n is None) else min(args.n, used)
        else:
            # if n isn't set, record the full train size for your output folder naming
            args.n = _len(dataset) if (args.n is None) else args.n

        # --- Pick reference from the test split matching your representation
        # Choose test file next to input_file: *_test_weighted.pt vs *_test.pt
        in_path = Path(args.input_file)
        if "weighted" in in_path.stem or "weighted" in in_path.name:
            test_path = in_path.with_name("mnist_test_weighted.pt")
        else:
            test_path = in_path.with_name("mnist_test_weighted.pt")

        test_data = DistributionPairDataset(str(test_path))

        # Extract reference pair (and weights if present)
        item0 = test_data[0]
        if isinstance(item0, (tuple, list)) and len(item0) == 4:
            mu_0, mu0_w, nu_0, nu0_w = item0
            regime = "mnist_weighted"
        else:
            mu_0, nu_0 = item0
            mu0_w = nu0_w = None
            regime = "mnist_sampled"

        # Identity baselines using the same representation as training/val
        def _as_weight(points, w):
            if w is None: return None
            w = torch.as_tensor(w, device=points.device, dtype=points.dtype).reshape(-1)
            s = w.sum()
            if (w.numel() != points.size(0)) or (s <= 0) or not torch.isfinite(w).all():
                return None
            return w / s

        def _id_base(example):
            if isinstance(example, (tuple, list)) and len(example) == 4:
                x, ax, y, ay = example
                ax = _as_weight(x, ax);
                ay = _as_weight(y, ay)
                if (ax is not None) and (ay is not None):
                    return float(sinkhorn_cost(x, y, ax, ay))  # <- wrapper handles order
                else:
                    return float(sinkhorn_cost(x, y))
            else:
                x, y = example
                return float(sinkhorn_cost(x, y))

        with torch.no_grad():
            base_train = _id_base(dataset[0])
            base_val   = _id_base(item0)

        print(f"[calib] train_id_base={base_train:.6f}  val_id_base={base_val:.6f}")

        # Build 1-item validation set in the correct tuple shape
        if mu0_w is not None:       # weighted
            val_list = [(mu_0, mu0_w, nu_0, nu0_w)]
        else:                        # sampled
            val_list = [(mu_0, nu_0)]

    else:
        dataset = DistributionPairDataset(args.input_file)
        mu_0, nu_0 = load_test_pair(args.k, args.dim)
        mu0_w = None
        base_train = base_val = None  # (optional: compute if you want ratios for GMM)
        val_list = [(mu_0, nu_0)]


    print("🔍 Computing bandwidth for kernel-weighted loss...", flush=True)
    # bw = knn_bandwidth(mu_0, mu_train, k=100)
    bw = 0.085
    print(f"🚀 Starting training (regime={args.regime}, seed={args.seed}, bw={bw:.4f})", flush=True)

    phase_dir = base_output_dir / f"mu0_seed{args.seed}_{args.mode}_n{args.n}_blur0.15"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Calling fit_local_map...", flush=True)
    start_time = time.time()

    result = fit_local_map(
        mu_0,
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
        # pass baselines (so ratios are consistent) and mu0 weights if any
        base_train=base_train,
        base_val=base_val,
        mu0_w=mu0_w,
    )

    val_losses = np.array(result["val_losses"])
    best_idx = int(val_losses.argmin())
    best_val = float(val_losses[best_idx])
    print(f"🏅 Best epoch = {best_idx} with val_loss = {best_val:.6g}")

    elapsed = time.time() - start_time
    print("✅ fit_local_map returned!", flush=True)
    print(f"⏱️ Training completed in {elapsed:.2f} seconds", flush=True)

    print("📊 Saving visualizations...", flush=True)

    if args.mode == "affine":
        for epoch, (alpha, B) in enumerate(zip(result["alpha_history"], result["B_history"])):
            if epoch % 1 == 0:
                visualize_pushforward(alpha, B, mu_0, nu_0, epoch, phase_dir)
    else:
        y_hist = result.get("y_hat_history", [])
        if isinstance(y_hist, (list, tuple)) and len(y_hist) > 0:
            for epoch, y_hat in enumerate(y_hist):
                if epoch % 7 == 0:
                    fig = plt.figure(figsize=(4, 4))
                    plt.scatter(mu_0[:, 1], mu_0[:, 0], s=2, label="μ₀")
                    plt.scatter(y_hat[:, 1], y_hat[:, 0], s=2, label="T#μ₀ (pred)")
                    plt.scatter(nu_0[:, 1], nu_0[:, 0], s=2, label="ν₀ (target)", alpha=0.6)
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.legend(loc="best", fontsize=8)
                    fig.savefig(phase_dir / f"pushforward_epoch_{epoch}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)

    loss_curves(result["train_losses"], result["val_losses"], phase_dir)

    # Overlay a vertical line at best epoch on the ratio plot
    loss_curves_with_ratio(
        result["train_losses"],
        result["val_losses"],
        result["base_train"],
        result["base_val"],
        result.get("lrs", []),
        phase_dir,
    )

    # If you want an explicit “best” text file:
    with open(phase_dir / "best_epoch.txt", "w") as f:
        f.write(f"{best_idx}\n")

    # Also visualize final pushforward for MNIST
    if args.regime.lower() == "mnist":
        if args.mode == "affine":
            # Use best alpha,B instead of last
            best_alpha = result["alpha_history"][best_idx]
            best_B = result["B_history"][best_idx]
            pred_pts = best_alpha + mu_0 @ best_B.T
        else:
            y_hist = result.get("y_hat_history", [])
            if isinstance(y_hist, (list, tuple)) and len(y_hist) > 0:
                pred_pts = y_hist[best_idx]  # <- best, not last
            else:
                pred_pts = result.get("final_pred", None)

        if pred_pts is not None:
            fig = plt.figure(figsize=(4, 4))
            plt.scatter(mu_0[:, 1], mu_0[:, 0], s=2, label="μ₀")
            plt.scatter(pred_pts[:, 1], pred_pts[:, 0], s=2, label="T#μ₀ (pred @ best)")
            plt.scatter(nu_0[:, 1], nu_0[:, 0], s=2, label="ν₀ (target)", alpha=0.6)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.legend(loc="best", fontsize=8)
            fig.savefig(phase_dir / "pushforward_best.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            def _get_meta(ds):
                base = getattr(ds, "dataset", ds)
                return getattr(base, "meta", {}) if hasattr(base, "meta") else {}

            meta = _get_meta(dataset)
            coord_range = meta.get("coord_range", "unit") if isinstance(meta, dict) else "unit"

            pred_img = pointcloud_to_image(pred_pts, w=mu0_w, coord_range=coord_range)
            target_img = pointcloud_to_image(nu_0, w=nu0_w, coord_range=coord_range)
            plt.imsave(phase_dir / "predicted_pushforward_img_best.png", pred_img.numpy(), cmap="gray")
            plt.imsave(phase_dir / "target_img.png", target_img.numpy(), cmap="gray")

    final_val_loss = result["val_losses"][-1]
    val_record = {
        "regime_id": args.regime,
        "mu0_id": 0,
        "dataset_idx": 0,
        "d": args.dim,
        "n": args.n,
        "k_samples": args.k,
        "gmm_components": None if args.regime.lower() == "mnist" else "TODO",
        "bw": bw,
        "val_loss": final_val_loss,
        "knn": 100,
        "train_time_sec": round(elapsed, 2),
        "map_seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_experiment(val_record, experiment_log_file)
    print(f"✅ Logged result: val_loss = {final_val_loss:.4f}\n", flush=True)

if __name__ == "__main__":
    main()
