import matplotlib.pyplot as plt
import torch


def plot_distribution_pair(mu_i, nu_i, index=None, show=True, save_path=None):
    """
    Plots a single pair (mu_i, nu_i) as scatter plots in 2D.
    """
    assert mu_i.shape[1] == 2, "Only supports 2D visualization."

    plt.figure(figsize=(5, 5))
    plt.scatter(mu_i[:, 0], mu_i[:, 1], alpha=0.4, label='mu_i', c='blue', s=10)
    plt.scatter(nu_i[:, 0], nu_i[:, 1], alpha=0.4, label='nu_i', c='red', s=10)
    plt.title(f"Distribution Pair {index}" if index is not None else "Distribution Pair")
    plt.legend()
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def visualize_pointcloud(pc, title="Point Cloud"):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(pc[:, 0], pc[:, 1], s=3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def visualize_pushforward(alpha, B, mu_i, nu_i, epoch, save_dir):
    transformed = (B @ mu_i.T).T + alpha

    plt.figure(figsize=(5, 5))
    plt.scatter(mu_i.detach().numpy()[:, 0], mu_i.detach().numpy()[:, 1], label='Source $\mu_i$', alpha=0.5)
    plt.scatter(nu_i.detach().numpy()[:, 0], nu_i.detach().numpy()[:, 1], label='Target $\nu_i$', alpha=0.5)
    plt.scatter(transformed.detach().numpy()[:, 0], transformed.detach().numpy()[:, 1], label='Transformed $T(\mu_i)$',
                alpha=0.5)
    plt.legend()
    plt.title(f"Evolution of $\mu_i \\to \\nu_i$ – Epoch {epoch}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"pushforward_epoch_{epoch}.png", dpi=150)
    plt.close()

def loss_curves(train_losses, val_losses, save_dir):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss – Local Affine Map")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_dir / f"loss_curve.png")
    # plt.show()
    plt.close()

@torch.no_grad()
def pointcloud_to_image(
    pts: torch.Tensor,
    w: torch.Tensor | None = None,
    H: int = 28,
    W: int = 28,
    coord_range: str = "unit",   # "unit" or "minus1_1"
    normalize: bool = True,      # scale to [0,1] for visualization
):
    """
    Rasterizes a point cloud into an HxW image.
    pts: (N, 2) with order (row, col).  If your coords are (y,x), this is correct.
    w  : optional (N,) nonnegative weights (sum arbitrary; normalized here)
    coord_range: "unit" -> coords in [0,1]; "minus1_1" -> coords in [-1,1].
    """
    assert pts.dim() == 2 and pts.size(1) == 2, f"pts shape must be (N,2), got {pts.shape}"

    pts = pts.to(torch.float32)
    device = pts.device

    # 1) Map coords to [0,1]
    if coord_range == "minus1_1":
        pts01 = (pts + 1.0) * 0.5
    elif coord_range == "unit":
        pts01 = pts
    else:
        raise ValueError(f"coord_range must be 'unit' or 'minus1_1', got {coord_range}")

    # Keep inside [0,1)
    eps = 1e-8
    pts01 = pts01.clamp(min=0.0, max=1.0 - eps)

    # 2) Convert to pixel indices (row, col)
    r = (pts01[:, 0] * (H - 1)).round().to(torch.long)
    c = (pts01[:, 1] * (W - 1)).round().to(torch.long)

    # 3) Weights
    if w is None:
        wv = torch.ones(pts.size(0), device=device, dtype=torch.float32)
    else:
        wv = torch.as_tensor(w, device=device, dtype=torch.float32).reshape(-1)
        if wv.numel() != pts.size(0):
            raise ValueError(f"weight length {wv.numel()} != number of points {pts.size(0)}")
        s = wv.sum()
        if s > 0:
            wv = wv / s

    # 4) Scatter-add into grid
    img = torch.zeros(H * W, device=device, dtype=torch.float32)
    idx = r * W + c
    img.scatter_add_(0, idx, wv)
    img = img.view(H, W)

    # 5) Normalize for display
    if normalize:
        m = img.max()
        if m > 0:
            img = img / m

    return img.cpu()

def _ema(xs, beta=0.9):
    if not xs: return []
    y = []
    m = xs[0]
    for v in xs:
        m = beta * m + (1 - beta) * v
        y.append(m)
    return y

def loss_curves_with_ratio(train_losses, val_losses, base_train, base_val, lrs, outdir, ema_beta=0.9):
    T = range(len(train_losses))
    tr = [t / max(base_train, 1e-12) for t in train_losses]
    vr = [v / max(base_val,   1e-12) for v in val_losses]

    # 1) Raw losses + EMA(val)
    plt.figure(figsize=(7,4))
    plt.plot(T, train_losses, label="Train")
    plt.plot(T, val_losses,   label="Val", alpha=0.35)
    plt.plot(T, _ema(val_losses, ema_beta), label=f"Val EMA β={ema_beta}", linewidth=2)
    plt.axhline(base_val, color='k', ls='--', lw=1, label="Val identity")
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("OT loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "loss_raw.png", dpi=160); plt.close()

    # 2) Ratios to identity (want < 1)
    plt.figure(figsize=(7,4))
    plt.plot(T, tr, label="Train / base_train")
    plt.plot(T, vr, label="Val / base_val", alpha=0.8)
    plt.axhline(1.0, color='k', ls='--', lw=1)
    plt.title("Improvement over identity (ratio)")
    plt.xlabel("Epoch"); plt.ylabel("× identity")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "loss_ratio.png", dpi=160); plt.close()

    # 3) Learning rate track
    if lrs is not None and len(lrs) == len(train_losses):
        plt.figure(figsize=(7,3))
        plt.step(T, lrs, where="post")
        plt.title("Learning rate")
        plt.xlabel("Epoch"); plt.ylabel("LR")
        plt.tight_layout()
        plt.savefig(outdir / "lr_schedule.png", dpi=160); plt.close()
