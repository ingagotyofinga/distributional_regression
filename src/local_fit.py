from torch.utils.data import DataLoader
import contextlib
from src.loss import kernel_weighted_loss, sinkhorn_cost
from src.estimator import build_estimator
from pathlib import Path
import torch, os, json, random
import numpy as np

def fit_local_map(
    mu_0,
    train_dataset,
    val_dataset,
    *,
    input_dim,
    hidden_dim=128,
    bw=0.5,
    lr=1e-4,
    n_epochs=100,
    batch_size=1,
    patience=10,                # <- scheduler patience
    phase_dir=None,
    verbose=True,
    mode="affine",
    residual_head='mono',
    viz_every=1,
    detect_anomaly=False,
    base_train=None,
    base_val=None,
    scheduler_threshold_rel=0.02,
    scheduler_cooldown=3,
    scheduler_min_lr=1e-6,
    scheduler_ema_beta=0.9,
    # NEW: optional reference weights for weighted MNIST
    mu0_w=None,
):
    """
    Train a local map centered at reference mu_0 using kernel-weighted loss.
    Returns histories plus the best validation and its identity ratio.
    """
    # Early stop is effectively disabled unless you change this:
    early_stop_patience = patience * 3  # TEST: no early stop
    lrs = []
    best_val = float('inf')
    best_epoch = -1
    wait = 0
    min_delta = 1e-6

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Model & optimizer
    model = build_estimator(
        arch="set_unet",  # or "mlp" for the original
        input_dim=input_dim,
        hidden_dim=128,  # try 128 or 256
        mode="residual",
        residual_head='monos'
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler: Reduce LR on plateau (with sensible settings for noisy curves)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=0.5,
        patience=patience,
        threshold=scheduler_threshold_rel,
        threshold_mode="rel",
        cooldown=scheduler_cooldown,
        min_lr=scheduler_min_lr,
        verbose=True,
    )

    def _rng_state():
        return {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    def save_ckpt_atomic(payload: dict, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    def save_checkpoint(tag: str, epoch: int, val_loss: float):
        if phase_dir is None:
            return
        payload = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": _rng_state(),
            "meta": {
                "tag": tag,
                "lr": float(next(iter(optimizer.param_groups))["lr"]) if optimizer is not None else None,
            },
        }
        save_ckpt_atomic(payload, phase_dir / f"{tag}.ckpt")

    def load_best_for_eval(model):
        if phase_dir is None:
            return None
        ckpt_path = phase_dir / "best.ckpt"
        if not ckpt_path.exists():
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        return ckpt

    # ---------- Identity baselines (calibration) ----------
    # Use weights iff the item provides them so ratios are apples-to-apples.
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

    # Logs / histories
    best_val, best_epoch = float("inf"), -1
    train_losses, val_losses = [], []
    y_hat_history = []                 # residual-mode snapshots for plotting
    alpha_history, B_history = [] , [] # filled only if affine

    anomaly_ctx = torch.autograd.detect_anomaly() if detect_anomaly else contextlib.nullcontext()
    val_ema = None  # smoothed metric for the scheduler

    phase_dir = Path(phase_dir) if phase_dir is not None else None
    if phase_dir is not None:
        phase_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        # -------- TRAIN --------
        model.train()
        tot_loss = 0.0
        tot_w_train = 0.0
        w_values = []  # collect kernel weights for ESS@e0

        for batch in train_loader:
            # Unpack batch: 4-tuple => weighted MNIST, 2-tuple => GMM/sampled
            if len(batch) == 4:
                mu_i, mu_w, nu_i, nu_w = batch
                active_regime = "mnist"
            else:
                mu_i, nu_i = batch
                mu_w = nu_w = None
                active_regime = "gmm"

            mu_i, nu_i = mu_i.squeeze(0), nu_i.squeeze(0)
            if mu_w is not None: mu_w = mu_w.squeeze(0)
            if nu_w is not None: nu_w = nu_w.squeeze(0)

            if mode == "affine":
                alpha, B = model.predict_params(mu_i, mu_w if mu_w is not None else None)
                loss, w = kernel_weighted_loss(
                    mu_i, nu_i, mu_0,
                    mode="affine", alpha=alpha, B=B,
                    bw=bw, return_normalizer=True,
                    regime=active_regime, mu_w=mu_w, nu_w=nu_w, mu0_w=mu0_w
                )
            else:  # residual / nonlinear
                out = model(mu_i, mu_w if mu_w is not None else None)
                y_hat = out[0] if isinstance(out, (tuple, list)) else out
                loss, w = kernel_weighted_loss(
                    mu_i, nu_i, mu_0,
                    mode="residual", pred=y_hat,
                    bw=bw, return_normalizer=True,
                    regime=active_regime, mu_w=mu_w, nu_w=nu_w, mu0_w=mu0_w
                )
                # print(f"Sinkhorn loss between mu and mu0:, "f"{sinkhorn_cost(mu_i, mu_0, mu_w if mu_w is not None else None, mu0_w if mu0_w is not None else None)}")

                # print(f"Training loss:, {loss}, Weight: {w}")

            optimizer.zero_grad()
            with anomaly_ctx:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss += float(loss)
            tot_w_train += float(w)
            w_values.append(float(w))

        # epoch train loss (normalized by total kernel weight)
        train_loss = tot_loss / max(tot_w_train, 1e-8)
        train_losses.append(train_loss)

        # --- kernel coverage snapshot once at epoch 0 ---
        if epoch == 0:
            eps = 1e-12
            weights = torch.tensor(w_values, dtype=torch.float32)
            kernel_sum = float(weights.sum())
            ess = float((weights.sum() ** 2) / (weights.pow(2).sum() + eps))
            q = torch.quantile(weights, torch.tensor([0.00, 0.25, 0.50, 0.75, 0.95]))
            print(f"[kernel@e0] n={weights.numel()} sum={kernel_sum:.4f} ESS={ess:.2f} "
                  f"q=[{q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f},{q[4]:.4f}]")
            for t in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
                cnt = int((weights > t).sum())
                print(f"[kernel@e0] count(w > {t:.3g}) = {cnt}")

        # -------- OPTIONAL AFFINE LOGGING AT REFERENCE --------
        if mode == "affine":
            with torch.no_grad():
                a_ref, B_ref = model.predict_params(mu_0, mu0_w if mu0_w is not None else None)
                alpha_history.append(a_ref.detach().cpu().clone())
                B_history.append(B_ref.detach().cpu().clone())

        # -------- RESIDUAL SNAPSHOT (every viz_every epochs) --------
        if mode != "affine" and (epoch % viz_every == 0):
            model.eval()
            with torch.no_grad():
                out_vis = model(mu_0, mu0_w if mu0_w is not None else None)
                y_vis = out_vis[0] if isinstance(out_vis, (tuple, list)) else out_vis
            y_hat_history.append(y_vis.detach().cpu().clone())

        # -------- VALIDATION --------
        model.eval()
        tot_v = 0.0
        tot_w_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    mu_i, mu_w, nu_i, nu_w = batch
                    active_regime = "mnist"
                else:
                    mu_i, nu_i = batch
                    mu_w = nu_w = None
                    active_regime = "gmm"

                mu_i, nu_i = mu_i.squeeze(0), nu_i.squeeze(0)
                if mu_w is not None: mu_w = mu_w.squeeze(0)
                if nu_w is not None: nu_w = nu_w.squeeze(0)

                if mode == "affine":
                    a, B = model.predict_params(mu_i, mu_w if mu_w is not None else None)
                    v, w = kernel_weighted_loss(
                        mu_i, nu_i, mu_0,
                        mode="affine", alpha=a, B=B,
                        bw=bw, return_normalizer=True,
                        regime=active_regime, mu_w=mu_w, nu_w=nu_w, mu0_w=mu0_w
                    )
                else:
                    out = model(mu_i, mu_w if mu_w is not None else None)
                    y_hat = out[0] if isinstance(out, (tuple, list)) else out
                    v, w = kernel_weighted_loss(
                        mu_i, nu_i, mu_0,
                        mode="residual", pred=y_hat,
                        bw=bw, return_normalizer=True,
                        regime=active_regime, mu_w=mu_w, nu_w=nu_w, mu0_w=mu0_w
                    )
                tot_v += float(v)
                tot_w_val += float(w)

        val_loss = tot_v / max(tot_w_val, 1e-8)
        # val_loss = tot_v
        val_losses.append(val_loss)

        # ---- Scheduler (on smoothed metric) ----
        val_ema = val_loss if val_ema is None else scheduler_ema_beta * val_ema + (1 - scheduler_ema_beta) * val_loss
        scheduler.step(val_ema)

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        # ---- Logging (ratios + current LR) ----
        _eps = 1e-12
        # just before computing ratios
        if base_val is None:
            base_val = 0.0  # or set to val_loss to be conservative
        # val_ratio = float(val_loss) / max(float(base_val), _eps)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch}] "
              f"Train: {train_loss:.6f} | "
              # f"Val: {val_loss:.6f} ({val_ratio:.6f}× id) | "
              f"Val: {val_loss:.6f} | "
              f"lr={current_lr:.2e}")

        # ---- Track best & optional early stop leash ----
        if phase_dir is not None:
            save_checkpoint(tag="last", epoch=epoch, val_loss=val_loss)

        if val_loss < best_val - min_delta:
            best_val, best_epoch = float(val_loss), int(epoch)
            if phase_dir is not None:
                save_checkpoint(tag="best", epoch=best_epoch, val_loss=best_val)
                # (Optional) keep a tiny human-readable sidecar
                with open(phase_dir / "best_meta.json", "w") as f:
                    json.dump({"best_val": best_val, "best_epoch": best_epoch}, f, indent=2)
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} "
                          f"(best @ epoch {best_epoch} with val={best_val:.6g})")
                break

    ckpt = load_best_for_eval(model)
    if ckpt is None and verbose:
        print("No best.ckpt found; using last in-memory weights.")
    else:
        if verbose:
            print(f"Loaded best checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6g}")

    # Final snapshot (residual mode)
    final_pred = None
    if mode != "affine":
        model.eval()
        with torch.no_grad():
            out_final = model(mu_0, mu0_w if mu0_w is not None else None)
            final_pred = (out_final[0] if isinstance(out_final, (tuple, list)) else out_final).detach().cpu().clone()

    # Return extras to help compare runs
    best_ratio = best_val / max(base_val, 1e-12)
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "alpha_history": alpha_history,
        "B_history": B_history,
        "y_hat_history": y_hat_history,
        "final_pred": final_pred,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_val_over_identity": best_ratio,
        "lrs": lrs,
        "base_train": base_train,
        "base_val": base_val,
    }
