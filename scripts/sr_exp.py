import os, random, math
import pandas as pd
import numpy as np
import json
import torch, torch.nn as nn
import torch.nn.functional as F
from pandas import concat
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from geomloss import SamplesLoss

# -----------------------
# Reproducibility
# -----------------------
SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------
# Device & constants
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 28

# -----------------------
# Pixel grid in [0,1]^2
# -----------------------
yy, xx = torch.meshgrid(
    torch.linspace(0,1,H, device=device),
    torch.linspace(0,1,W, device=device),
    indexing="ij"
)
X = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (784, 2)

# Sinkhorn W2^2 on the grid (balanced)
BLUR = 0.0075
sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=BLUR)  # ~1–2 px on [0,1]

def _to_probs(img):
    img = torch.nn.functional.softplus(img)             # nonnegative
    return img / (img.sum(dim=(2,3), keepdim=True) + 1e-12)

# Per-sample W2^2 (loop is fine for MNIST)
@torch.no_grad()
def w2_batch_nograd(a, b):
    vals = []
    for ai, bi in zip(a, b):
        ai = _to_probs(ai.unsqueeze(0)).view(-1)        # (784,)
        bi = _to_probs(bi.unsqueeze(0)).view(-1)
        vals.append(sinkhorn(ai, X, bi, X))
    return torch.stack(vals)                             # (B,)

def w2_batch(a, b):
    vals = []
    for ai, bi in zip(a, b):
        ai = _to_probs(ai.unsqueeze(0)).view(-1)
        bi = _to_probs(bi.unsqueeze(0)).view(-1)
        vals.append(sinkhorn(ai, X, bi, X))
    return torch.stack(vals)

# Gaussian kernel on W2^2
def kernel_weights(mu0, mu_batch, h=0.01):
    with torch.no_grad():
        d2 = w2_batch_nograd(mu_batch, mu0.expand_as(mu_batch))
    return torch.exp(-d2 / (2 * h * h))                  # (B,)

# -----------------------
# Model
# -----------------------
def conv3x3(in_ch, out_ch, stride=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )

class ResBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
        )
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.net(x) + x)

class TinyUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        c1, c2 = base, base*2

        # Encoder
        self.e1 = nn.Sequential(conv3x3(1, c1), ResBlock(c1))           # 28x28
        self.e2 = nn.Sequential(
            conv3x3(c1, c2, stride=2),                                   # 14x14
            ResBlock(c2)
        )

        # Bottleneck (still 14x14 to keep it small)
        self.mid = nn.Sequential(conv3x3(c2, c2), ResBlock(c2))

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear")  # 14→28
        self.d1  = nn.Sequential(
            conv3x3(c2 + c1, c1),                                        # concat skip from e1
            ResBlock(c1)
        )

        # Head
        self.out = nn.Conv2d(c1, 1, 1)                                   # linear; softplus in loss

    def forward(self, x):
        s1 = self.e1(x)            # 28x28, c1
        s2 = self.e2(s1)           # 14x14, c2
        z  = self.mid(s2)          # 14x14, c2
        y  = self.up1(z)           # 28x28, c2
        y  = torch.cat([y, s1], dim=1)  # 28x28, c2+c1
        y  = self.d1(y)            # 28x28, c1
        return self.out(y)

def to_display_img(yhat, normalize_mass=True):
    img = torch.nn.functional.softplus(yhat)
    if normalize_mass:
        img = img / (img.sum(dim=(2,3), keepdim=True) + 1e-12)
    mn = img.amin(dim=(2,3), keepdim=True)
    mx = img.amax(dim=(2,3), keepdim=True)
    disp = (img - mn) / (mx - mn + 1e-12)
    return disp.clamp(0,1)

# -----------------------
# Data: Super-resolution (LR-upsampled -> HR)
# -----------------------
train = MNIST(root=".", train=True, download=True, transform=ToTensor())

# Super-resolution degradation settings
SR_SCALE = 2                 # 2x SR: 28->14->28 (try 4 for 28->7->28)
DEGR_BLUR_SIGMA = 1.0        # set to None or 0.0 to disable pre-blur
UPSAMPLE_MODE = "bilinear"   # "bilinear" (or "bicubic" if you prefer)
THRESH = 1e-5
h_kernel = 0.0019

# How many MNIST samples to use total (across all digits)
N_TOTAL = 1000               # keep modest; OT is expensive
N_TEST = 10
MU0_LABEL = 5

def degrade_to_lr_up(y_hr, *, scale=2, blur_sigma=1.0, up_mode="bilinear"):
    # Create an LR input by blur->downsample->upsample back to 28x28.
    # y_hr: (N,1,28,28) in [0,1]
    y = y_hr
    if blur_sigma is not None and blur_sigma > 0:
        # kernel size ~ 6*sigma + 1, rounded to odd
        k = int(6 * blur_sigma + 1)
        if k % 2 == 0:
            k += 1
        y = TF.gaussian_blur(y, kernel_size=[k, k], sigma=[blur_sigma, blur_sigma])

    h_lr = H // scale
    w_lr = W // scale

    # Downsample with area averaging (anti-aliased)
    x_lr = F.interpolate(y, size=(h_lr, w_lr), mode="area")

    # Upsample back to HR grid for U-Net input
    x_lr_up = F.interpolate(
        x_lr, size=(H, W), mode=up_mode,
        align_corners=False if up_mode in ("bilinear", "bicubic") else None
    )
    return x_lr_up

def collect_any(ds, n):
    xs, ys = [], []
    for img, y in ds:
        xs.append(img)            # (1,28,28)
        ys.append(int(y))
        if len(xs) >= n:
            break
    X = torch.stack(xs, dim=0)               # (N,1,28,28)
    y = torch.tensor(ys, dtype=torch.long)   # (N,)
    return X, y

# HR targets (CPU)
y_hr_all, labels_all = collect_any(train, N_TOTAL)

# LR inputs (CPU) generated from HR
x_lr_up_all = degrade_to_lr_up(
    y_hr_all, scale=SR_SCALE, blur_sigma=DEGR_BLUR_SIGMA, up_mode=UPSAMPLE_MODE
)

# ------------------------------------------------------------
# Choose reference (mu0, nu0) by DIGIT LABEL (not by index)
# ------------------------------------------------------------
same_digit_all = torch.where(labels_all == MU0_LABEL)[0]
if same_digit_all.numel() == 0:
    raise ValueError(f"No samples with label {MU0_LABEL} found in the first N_TOTAL={N_TOTAL} examples.")

mu0_idx = same_digit_all[0].item()  # pick the first one (or randomize if you prefer)
mu0 = x_lr_up_all[mu0_idx:mu0_idx+1].to(device)   # (1,1,28,28)
nu0 = y_hr_all[mu0_idx:mu0_idx+1].to(device)

# ------------------------------------------------------------
# Build SAME-DIGIT local test set supported by kernel
# ------------------------------------------------------------
cand_idx = same_digit_all[same_digit_all != mu0_idx]   # dataset indices (CPU)
cand_mu = x_lr_up_all[cand_idx].to(device)             # (B,1,28,28)

# ---- kernel support filter (vectorized) ----
w = kernel_weights(mu0, cand_mu, h=h_kernel)               # (B,)
supported_idx = cand_idx[w >= THRESH]                  # dataset indices (CPU)

if supported_idx.numel() < N_TEST:
    raise ValueError(
        f"Only {supported_idx.numel()} supported samples for digit {MU0_LABEL} "
        f"with THRESH={THRESH}. Increase N_TOTAL or lower THRESH / increase h."
    )

# ---- random select N_TEST supported indices ----
perm = torch.randperm(supported_idx.numel())
selected_indices = supported_idx[perm[:N_TEST]]        # dataset indices (CPU)

with torch.no_grad():
    d2_test_to_mu0 = w2_batch_nograd(x_lr_up_all[selected_indices].to(device), mu0.expand(N_TEST,1,H,W))
print("test d2(mu_i, mu0):", d2_test_to_mu0.detach().cpu().numpy())
print("min/mean/max:", float(d2_test_to_mu0.min()), float(d2_test_to_mu0.mean()), float(d2_test_to_mu0.max()))

# ---- build test set ----
src_test = x_lr_up_all[selected_indices]               # CPU (move later if you want)
tgt_test = y_hr_all[selected_indices]
src_test_labels = labels_all[selected_indices]
tgt_test_labels = labels_all[selected_indices]
src_test = src_test.to(device)
tgt_test = tgt_test.to(device)

# ------------------------------------------------------------
# Training pool = all indices excluding mu0 + excluding test set
# ------------------------------------------------------------
all_indices = torch.arange(N_TOTAL, dtype=torch.long)
exclude = torch.zeros(N_TOTAL, dtype=torch.bool)
exclude[mu0_idx] = True
exclude[selected_indices] = True

train_indices = all_indices[~exclude]

src = x_lr_up_all[train_indices]
tgt = y_hr_all[train_indices]
src_labels = labels_all[train_indices]
tgt_labels = labels_all[train_indices]

# Move full pool to device (kernel + model on same device)
src = src.to(device)
tgt = tgt.to(device)

# DataLoader keeps CPU tensors; we'll .to(device) inside the loop
M = src.shape[0]
indices = torch.arange(M)  # indices into THIS TRAIN POOL (0..M-1)
dataset = TensorDataset(src.cpu(), tgt.cpu(), indices)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

#------------------------
# Diagnostic
#------------------------

@torch.no_grad()
def kernel_neighbor_report(
    mu0,                   # (1,1,28,28)
    src,                   # (N,1,28,28)
    src_labels,            # (N,) long
    *,
    h=0.0045,              # your bandwidth / blur
    topk=32,
    tau=0.01,              # threshold on kernel weight
    kernel_fn=None,        # optional; if None, uses your project’s kernel_weights
    device=None,
    title="kernel@mu0"
):
    """
    Prints label stats for neighbors 'caught' by the kernel around mu0.
    Returns a dict with raw tensors for further logging if you want.
    """
    device = device or (mu0.device if hasattr(mu0, "device") else "cpu")
    mu0 = mu0.to(device)
    src = src.to(device)
    src_labels = src_labels.to(device)

    # 1) weights
    w = kernel_weights(mu0, src, h=h)        # (N,)

    w = w.reshape(-1)
    N = w.numel()

    # 2) Top-K neighbors by weight
    k = min(topk, N)
    tk = torch.topk(w, k=k, largest=True, sorted=True)
    idx_top = tk.indices
    w_top = tk.values
    y_top = src_labels[idx_top]

    print(f"\n[{title}] Top-{k} neighbors by kernel weight")
    for rank, (i, wi, yi) in enumerate(zip(idx_top.tolist(), w_top.tolist(), y_top.tolist()), 1):
        print(f"  #{rank:02d}: idx={i:5d}  label={yi}  w={wi:.6f}")

    # 3) Thresholded neighbors
    idx_tau = torch.nonzero(w >= tau, as_tuple=False).squeeze(-1)
    y_tau = src_labels[idx_tau]
    print(f"\n[{title}] Neighbors with w >= {tau} : count={idx_tau.numel()}")
    if idx_tau.numel() > 0:
        counts = torch.bincount(y_tau, minlength=10)  # MNIST 0..9
        print("  Label counts above tau:", counts.tolist())

    # 4) Weighted label histogram (+ normalized)
    #    This is the *most* telling number: where is the kernel mass going?
    weights_per_label = torch.zeros(10, device=device)
    weights_per_label.index_add_(0, src_labels, w)
    total_w = w.sum().item()
    hist = weights_per_label.cpu().numpy()
    hist_norm = (hist / (total_w + 1e-12)).tolist()
    print(f"\n[{title}] Weighted label mass (sum w = {total_w:.6f})")
    print("  mass per label:       ", [round(x, 6) for x in hist])
    print("  normalized mass (pdf):", [round(x, 6) for x in hist_norm])

    # 5) Effective Sample Size (ESS)
    ess = (total_w**2) / (w.pow(2).sum().item() + 1e-12)
    print(f"\n[{title}] ESS = {ess:.2f} / N={N}  (higher ≈ flatter kernel)")

    # 6) OPTIONAL: quick viz of the top neighbors
    try:
        ncols = 8
        nrows = math.ceil(k / ncols)
        plt.figure(figsize=(ncols*1.2, nrows*1.2))
        for j, idx in enumerate(idx_top.tolist(), 1):
            plt.subplot(nrows, ncols, j)
            plt.imshow(src[idx].squeeze().cpu(), cmap="gray")
            plt.axis("off")
            plt.title(f"y={int(src_labels[idx])}\nw={float(w[idx]):.3g}", fontsize=8)
        plt.suptitle(f"{title}: Top-{k} neighbors")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(viz skipped: {e})")

    return {
        "w": w,
        "idx_top": idx_top,
        "w_top": w_top,
        "y_top": y_top,
        "idx_tau": idx_tau,
        "y_tau": y_tau,
        "weights_per_label": weights_per_label,
        "ess": ess,
        "total_w": total_w,
    }


# -----------------------
# Train config
# -----------------------


lr = 1e-3
n_epochs = 20
base = 32

# Super-resolution tip: residual prediction often stabilizes training
RESIDUAL_PRED = True  # nu_hat = x + model(x)

report = kernel_neighbor_report(
    mu0, src, src_labels, h=h_kernel, topk=10, tau=0.01, title="kernel@mu0"
)

pair = f"SR_x{SR_SCALE}_digit{MU0_LABEL}"
run_name = f"{pair}_FULL_epochs{n_epochs}_h{h_kernel:.5f}_bl{BLUR:.4f}_b{base}_lr{lr:.0e}"
results_dir = os.path.join("../results/mnist/sr",run_name)
os.makedirs(results_dir, exist_ok=True)

meta = {
    "SEED": SEED,
    "device": device,
    "BLUR": BLUR,
    "h_kernel": h_kernel,
    "THRESH": THRESH,
    "SR_SCALE": SR_SCALE,
    "DEGR_BLUR_SIGMA": DEGR_BLUR_SIGMA,
    "UPSAMPLE_MODE": UPSAMPLE_MODE,
    "N_TOTAL": N_TOTAL,
    "N_TEST": N_TEST,
    "MU0_LABEL": MU0_LABEL,
    "mu0_idx": int(mu0_idx),
    "test_indices": [int(i) for i in selected_indices.cpu().tolist()],
    "train_indices": [int(i) for i in train_indices.cpu().tolist()],
}
with open(os.path.join(results_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# -----------------------
# Logging + metrics output
# -----------------------
log_path = os.path.join(results_dir, "train_log.txt")
csv_path = os.path.join(results_dir, "metrics.csv")

def logprint(msg: str):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

metrics_rows = []  # list of dicts, one per epoch

def write_metrics_csv(rows):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df

save_image(to_display_img(mu0, normalize_mass=True).detach().cpu(),
           os.path.join(results_dir, f"mu0_{pair}.png"), nrow=1)
save_image(to_display_img(nu0, normalize_mass=True).detach().cpu(),
           os.path.join(results_dir, f"nu0_{pair}.png"), nrow=1)

model = TinyUNet(base=base).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

# ---- Best checkpoint (minimal) ----
best_val = float("inf")
best_epoch = -1
best_model_path = os.path.join(results_dir, "best.pt")
best_img_path   = os.path.join(results_dir, "pred_mu0_to_nu0hat_best.png")

loss_history = []
ema_history = []
ema = None

for epoch in range(n_epochs):
    model.train()

    # Precompute weights once per epoch against μ0 and normalize
    all_w = kernel_weights(mu0, src, h=h_kernel).detach()   # (M,)
    W = float(all_w.sum().cpu()) + 1e-12
    ess = (W ** 2) / float((all_w ** 2).sum().cpu())
    cnt01 = int((all_w > 0.1).sum().cpu())
    cnt005 = int((all_w > 0.05).sum().cpu())
    logprint(f"[kernel@init] ESS={ess:.1f}/{M}  sum={W:.6f}  count(w>0.1)={cnt01}  count(w>0.05)={cnt005}")

    epoch_num = 0.0

    for (x_i, y_i, idx_i) in loader:
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        # each is shape (1,1,28,28); idx_i is shape (1,)
        w_i = all_w[idx_i.item()]          # scalar tensor on device
        yhat_i = (x_i + model(x_i)) if RESIDUAL_PRED else model(x_i)

        d2_i = w2_batch(yhat_i, y_i)[0]    # scalar
        loss_i = (w_i / W) * d2_i          # unbiased SGD on normalized objective

        opt.zero_grad()
        loss_i.backward()
        opt.step()

        # accumulate numerator for epoch-mean (same normalized objective)
        epoch_num += float((w_i * d2_i).detach().cpu())

    loss_epoch = epoch_num / W
    loss_history.append(loss_epoch)

    # EMA for readability
    ema = loss_epoch if ema is None else 0.9 * ema + 0.1 * loss_epoch
    ema_history.append(ema)

    # Local validation on SAME-DIGIT nearby test set
    model.eval()
    with torch.no_grad():
        pred_test = (src_test + model(src_test)) if RESIDUAL_PRED else model(src_test)
        d2_test = w2_batch_nograd(pred_test, tgt_test)  # (N_TEST,)

        val_w2_mean = float(d2_test.mean().cpu())
        val_w2_std = float(d2_test.std(unbiased=False).cpu())
        val_w2_min = float(d2_test.min().cpu())
        val_w2_max = float(d2_test.max().cpu())

    val_metric = val_w2_mean
    # ---- Update best and save preview/model ----
    if val_metric < best_val:
        best_val = val_metric
        best_epoch = epoch
        torch.save({"epoch": epoch, "model": model.state_dict()}, best_model_path)

        # Save a grid preview of test predictions (handy for writeup)
        disp_grid = to_display_img(pred_test, normalize_mass=True).detach().cpu()
        save_image(disp_grid, os.path.join(results_dir, f"pred_test_best_digit{MU0_LABEL}.png"),
                   nrow=min(5, pred_test.shape[0]))

        # Save the corresponding LR-up inputs and HR targets for comparison
        disp_src = to_display_img(src_test, normalize_mass=True).detach().cpu()
        disp_tgt = to_display_img(tgt_test, normalize_mass=True).detach().cpu()

        save_image(disp_src, os.path.join(results_dir, f"test_src_lr_up_digit{MU0_LABEL}.png"),
                   nrow=min(5, src_test.shape[0]))
        save_image(disp_tgt, os.path.join(results_dir, f"test_tgt_hr_digit{MU0_LABEL}.png"),
                   nrow=min(5, tgt_test.shape[0]))

        # (Optional) also keep the mu0->nu0 preview if you still like it:
        nu0hat = (mu0 + model(mu0)) if RESIDUAL_PRED else model(mu0)
        disp_best = to_display_img(nu0hat, normalize_mass=True).detach().cpu()
        save_image(disp_best, best_img_path, nrow=1)

        logprint(f"[ckpt] saved best @ epoch {epoch}  test_mean={val_w2_mean:.6f}")

    metrics_rows.append({
        "epoch": epoch,
        "train_loss_epoch": loss_epoch,
        "train_ema": ema,

        # kernel diagnostics on training pool
        "kernel_sumW": W,
        "kernel_ess": ess,
        "kernel_cnt_w_gt_0p1": cnt01,
        "kernel_cnt_w_gt_0p05": cnt005,

        # test metrics (local same-digit test set)
        "test_w2_mean": val_w2_mean,
        "test_w2_std": val_w2_std,
        "test_w2_min": val_w2_min,
        "test_w2_max": val_w2_max,

        # bookkeeping
        "best_val_so_far": best_val,
        "best_epoch_so_far": best_epoch,
    })

    # write every epoch so you never lose progress if it crashes
    _ = write_metrics_csv(metrics_rows)

    logprint(
        f"epoch {epoch:03d}: train={loss_epoch:.6f} | ema={ema:.6f} | "
        f"test_mean={val_w2_mean:.6f} | std={val_w2_std:.6f} | "
        f"min={val_w2_min:.6f} | max={val_w2_max:.6f}"
    )
# -----------------------
# Save preview + loss curve (final weights + curve)
# -----------------------
df = pd.read_csv(csv_path)

# Train curves
plt.figure()
plt.plot(df["epoch"], df["train_loss_epoch"], label="train_loss_epoch")
plt.plot(df["epoch"], df["train_ema"], label="train_ema")
plt.xlabel("epoch"); plt.ylabel("train objective")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curve.png"))
plt.close()

# Test curves
plt.figure()
plt.plot(df["epoch"], df["test_w2_mean"], label="test_w2_mean")
plt.fill_between(df["epoch"],
                 df["test_w2_mean"] - df["test_w2_std"],
                 df["test_w2_mean"] + df["test_w2_std"],
                 alpha=0.2, label="±1 std")
plt.xlabel("epoch"); plt.ylabel("test W2^2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "val_curve.png"))
plt.close()

# (Optional) also reload the best weights and resave the preview to be sure:
if os.path.exists(best_model_path):
    chkpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(chkpt["model"])
    with torch.no_grad():
        nu0hat_best = (mu0 + model(mu0)) if RESIDUAL_PRED else model(mu0)
    disp_best = to_display_img(nu0hat_best, normalize_mass=True).detach().cpu()
    save_image(disp_best, best_img_path, nrow=1)
