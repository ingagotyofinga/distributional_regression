# =======================
# MNIST local OT regression (2→8) with distance cache + top-p trimming
# (fixed robust distance caching: memory+disk, no KeyError, clear HIT/MISS logs)
# =======================
import os, random, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
import hashlib

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

CACHE_DIR = Path("./_ot_cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Pixel grid in [0,1]^2
# -----------------------
yy, xx = torch.meshgrid(
    torch.linspace(0,1,H, device=device),
    torch.linspace(0,1,W, device=device),
    indexing="ij"
)
X = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (784, 2)

# -----------------------
# Sinkhorn W2^2 on the grid (balanced)
# NOTE: If you change BLUR, recreate 'sinkhorn' and clear DIST_CACHE.
# -----------------------
BLUR = 0.04
sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=BLUR)

# Optional helper to change blur in pixel units and clear cache
DIST_CACHE = {}  # key: (mu_tag, blur_float, src_tag) -> (N,) d2

def set_blur_pixels(px: float):
    """Set blur in *pixel* units and clear distance cache."""
    global BLUR, sinkhorn, DIST_CACHE
    BLUR = float(px) / 28.0
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=BLUR)
    DIST_CACHE.clear()

# -----------------------
# Prob helpers & batched W2
# -----------------------

def _to_probs(img):
    img = torch.nn.functional.softplus(img)
    return img / (img.sum(dim=(2,3), keepdim=True) + 1e-12)

@torch.no_grad()
def images_to_prob_rows(img_batch):
    x = torch.nn.functional.softplus(img_batch)
    x = x / (x.sum(dim=(2,3), keepdim=True) + 1e-12)
    return x.view(x.size(0), -1).contiguous()

# Per-sample W2^2 with grad (used in training)

def w2_batch(a, b):
    vals = []
    for ai, bi in zip(a, b):
        ai = _to_probs(ai.unsqueeze(0)).view(-1)
        bi = _to_probs(bi.unsqueeze(0)).view(-1)
        vals.append(sinkhorn(ai, X, bi, X))
    return torch.stack(vals)

# -----------------------
# Model (tiny U-Net)
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
        self.e1 = nn.Sequential(conv3x3(1, c1), ResBlock(c1))
        self.e2 = nn.Sequential(conv3x3(c1, c2, stride=2), ResBlock(c2))
        self.mid = nn.Sequential(conv3x3(c2, c2), ResBlock(c2))
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d1  = nn.Sequential(conv3x3(c2 + c1, c1), ResBlock(c1))
        self.out = nn.Conv2d(c1, 1, 1)
    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        z  = self.mid(s2)
        y  = self.up1(z)
        y  = torch.cat([y, s1], dim=1)
        y  = self.d1(y)
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
# Data (0→5, 1→7, 2→8, 3→6, 4→9)
# -----------------------
train = MNIST(root=".", train=True, download=True, transform=ToTensor())

N_PER_CLASS = 1000
pairs = [(0,5),(1,7),(2,8),(3,6),(4,9)]

def collect(ds, label, n=None):
    xs, ys = [], []
    for img, y in ds:
        if int(y) == label:
            xs.append(img)
            ys.append(label)
            if n is not None and len(xs) >= n:
                break
    Xb = torch.stack(xs, dim=0)
    yb = torch.tensor(ys, dtype=torch.long)
    return Xb, yb

mu0s, nu0s = {}, {}
src_chunks, tgt_chunks = [], []
src_label_chunks, tgt_label_chunks = [], []
for s, t in pairs:
    src_all,src_all_labels = collect(train, s, N_PER_CLASS)
    tgt_all,tgt_all_labels = collect(train, t, N_PER_CLASS)
    mu0s[(s,t)] = src_all[:1].to(device)
    nu0s[(s,t)] = tgt_all[:1].to(device)
    src_chunks.append(src_all[1:])
    tgt_chunks.append(tgt_all[1:])
    src_label_chunks.append(src_all_labels[1:])
    tgt_label_chunks.append(tgt_all_labels[1:])

src = torch.cat(src_chunks, dim=0).to(device)
tgt = torch.cat(tgt_chunks, dim=0).to(device)
src_labels = torch.cat(src_label_chunks, dim=0)
tgt_labels = torch.cat(tgt_label_chunks, dim=0)
assert src.shape[0] == src_labels.shape[0]
assert tgt.shape[0] == tgt_labels.shape[0]
M = src.shape[0]

# -----------------------
# Choose the mapping and reference
# -----------------------
pair = (4,9)
mu0 = mu0s[pair]
nu0 = nu0s[pair]

# ======================================================
# Robust squared W2 distance cache + weights + trimming
# ======================================================

@torch.no_grad()
def tensor_fingerprint(t: torch.Tensor, max_bytes: int = 1_000_000) -> str:
    """MD5 of up to max_bytes of tensor data + shape + dtype."""
    t = t.detach().contiguous().cpu()
    h = hashlib.md5()
    h.update(str(t.shape).encode()); h.update(str(t.dtype).encode())
    view = t.view(-1).to(torch.float32)
    n = min(view.numel(), max_bytes // 4)
    if n > 0:
        h.update(view[:n].numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def compute_squared_w2_to_mu0_cached(mu0, src, *, mu0_key="default", chunk_size=256):
    """Return d2[i] = W2^2(src[i], mu0) using current global 'sinkhorn' (BLUR).
    Uses in‑memory and on‑disk caches. Prints clear HIT/MISS logs.
    """
    # Strong content‑based keys (change if mu0 or src ORDER/contents change)
    mu_tag  = f"{mu0_key}-{tensor_fingerprint(mu0)}"
    src_tag = tensor_fingerprint(src)
    blur_f  = float(BLUR)
    key     = (mu_tag, blur_f, src_tag)

    # 1) Try memory cache
    d2 = DIST_CACHE.get(key, None)
    if d2 is not None:
        print("[d2 cache] HIT (memory)")
        return d2

    # 2) Try disk cache
    disk_name = f"d2_mu[{mu_tag}]_bl[{blur_f:.6f}]_src[{src_tag}].pt"
    disk_path = CACHE_DIR / disk_name
    if disk_path.exists():
        print(f"[d2 cache] HIT (disk) -> {disk_path}")
        d2 = torch.load(str(disk_path), map_location=mu0.device)
        DIST_CACHE[key] = d2
        return d2

    # 3) MISS → compute distances
    print("[d2 cache] MISS → computing distances …")
    mu0_probs = images_to_prob_rows(mu0)     # (1,784)
    src_probs = images_to_prob_rows(src)     # (N,784)

    vals = []
    N = src_probs.size(0)
    for i in range(0, N, chunk_size):
        a = src_probs[i:i+chunk_size]
        b = mu0_probs.expand(a.size(0), -1)
        for ai, bi in zip(a, b):
            vals.append(sinkhorn(ai, X, bi, X))
    d2 = torch.stack(vals).reshape(-1)

    # Save both caches
    DIST_CACHE[key] = d2.detach().clone()
    torch.save(DIST_CACHE[key].cpu(), str(disk_path))
    print(f"[d2 cache] STORED -> {disk_path}")
    return DIST_CACHE[key]

@torch.no_grad()
def gaussian_kernel_weights_from_squared_distances(d2, h):
    return torch.exp(-d2 / (2.0 * h * h))

@torch.no_grad()
def effective_sample_size(w):
    s = w.sum()
    return float((s * s) / (w.pow(2).sum().item() + 1e-12))

@torch.no_grad()
def trim_top_p(w, labels, p=0.95):
    vals, idx = torch.sort(w, descending=True)
    cum = torch.cumsum(vals, dim=0)
    k = int((cum <= p * cum[-1]).sum().item())
    k = max(k, 1)
    keep_idx = idx[:k]
    wk = w[keep_idx] / (w[keep_idx].sum() + 1e-12)
    mass = torch.zeros(10, device=w.device)
    mass.index_add_(0, labels[keep_idx], wk)
    purity = float(mass.max().item())
    ess = effective_sample_size(wk)
    return keep_idx, wk, purity, ess

@torch.no_grad()
def trim_by_threshold(w, labels, tau=0.008):
    mask = (w >= tau)
    keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    wk = w[keep_idx] / (w[keep_idx].sum() + 1e-12)
    mass = torch.zeros(10, device=w.device)
    mass.index_add_(0, labels[keep_idx], wk)
    purity = float(mass.max().item())
    ess = effective_sample_size(wk)
    return keep_idx, wk, purity, ess

# -----------------------
# Neighbor report (from cached d2)
# -----------------------
@torch.no_grad()
def kernel_neighbor_report_from_d2(
    d2, src, src_labels, *, h, topk=32, tau=0.01, title="kernel@mu0(d2)"
):
    w = gaussian_kernel_weights_from_squared_distances(d2, h).reshape(-1)
    N = w.numel()
    k = min(topk, N)
    tk = torch.topk(w, k=k, largest=True, sorted=True)
    idx_top = tk.indices
    w_top = tk.values
    y_top = src_labels[idx_top]
    print(f"\n[{title}] Top-{k} neighbors by kernel weight")
    for rank, (i, wi, yi) in enumerate(zip(idx_top.tolist(), w_top.tolist(), y_top.tolist()), 1):
        print(f"  #{rank:02d}: idx={i:5d}  label={yi}  w={wi:.6f}")
    idx_tau = torch.nonzero(w >= tau, as_tuple=False).squeeze(-1)
    y_tau = src_labels[idx_tau]
    print(f"\n[{title}] Neighbors with w >= {tau} : count={idx_tau.numel()}")
    if idx_tau.numel() > 0:
        counts = torch.bincount(y_tau, minlength=10)
        print("  Label counts above tau:", counts.tolist())
    weights_per_label = torch.zeros(10, device=w.device)
    weights_per_label.index_add_(0, src_labels, w)
    total_w = w.sum().item()
    hist = weights_per_label.detach().cpu().numpy()
    hist_norm = (hist / (total_w + 1e-12)).tolist()
    print(f"\n[{title}] Weighted label mass (sum w = {total_w:.6f})")
    print("  mass per label:       ", [round(x, 6) for x in hist])
    print("  normalized mass (pdf):", [round(x, 6) for x in hist_norm])
    ess = effective_sample_size(w)
    print(f"\n[{title}] ESS = {ess:.2f} / N={N}  (higher ≈ flatter kernel)")
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
    return {"w": w, "idx_top": idx_top, "w_top": w_top, "y_top": y_top,
            "idx_tau": idx_tau, "y_tau": y_tau,
            "weights_per_label": weights_per_label, "ess": ess, "total_w": total_w}

# -----------------------
# Workflow
# -----------------------

# 1) Cache squared W2 distances once for this mu0 + BLUR
d2 = compute_squared_w2_to_mu0_cached(mu0, src, mu0_key="2->8_mu0#0", chunk_size=256)

# 2) Choose bandwidth manually
h_kernel = 0.0022

# 3) Report (fast; uses cached d2)
_ = kernel_neighbor_report_from_d2(
    d2, src, src_labels,
    h=h_kernel, topk=32, tau=0.01,
    title=f"kernel@mu0(d2) BLUR={BLUR} h={h_kernel}"
)

# 4) Top-p trimming
top_p = 0.85
w_all = gaussian_kernel_weights_from_squared_distances(d2, h_kernel)
keep_idx, w_kept, purity, ess_kept = trim_top_p(w_all, src_labels, p=top_p)
print(f"[trim-top-p] kept={len(keep_idx)}  purity={purity:.3f}  ESS_after_trim={ess_kept:.2f}")

# ========= TRAIN + VIZ (trimmed neighborhood) =========

src_eff = src[keep_idx]
tgt_eff = tgt[keep_idx]
w_eff   = w_kept.clone()

alpha = 1.6   # try 1.4–1.8
w_eff = (w_kept ** alpha)
w_eff = w_eff / (w_eff.sum() + 1e-12)

base      = 32
lr        = 1e-3
n_epochs  = 200
save_every = 10

run_name   = f"{pair}_FULL_epochs{n_epochs}_h{h_kernel:.5f}_bl{BLUR:.3f}_b{base}_lr{lr:.0e}_topp{top_p:.2f}"
results_dir = os.path.join("../results/mnist", run_name)
os.makedirs(results_dir, exist_ok=True)

save_image(mu0.detach().cpu(), os.path.join(results_dir, f"mu0_{pair}.png"))
save_image(nu0.detach().cpu(), os.path.join(results_dir, f"nu0_{pair}.png"))

model = TinyUNet(base=base).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

best_val        = float("inf")
best_epoch      = -1
best_model_path = os.path.join(results_dir, "best.pt")
best_img_path   = os.path.join(results_dir, "pred_mu0_to_nu0hat_best.png")

loss_history = []
ema_history  = []
ema_beta     = 0.9
ema_value    = None

for epoch in range(1, n_epochs + 1):
    model.train()
    yhat = model(src_eff)
    d2_eff = w2_batch(yhat, tgt_eff)
    loss = (w_eff * d2_eff).sum()
    opt.zero_grad(); loss.backward(); opt.step()

    loss_val = float(loss.detach().cpu())
    loss_history.append(loss_val)
    ema_value = loss_val if ema_value is None else (ema_beta * ema_value + (1 - ema_beta) * loss_val)
    ema_history.append(ema_value)

    model.eval()
    with torch.no_grad():
        nu0hat = model(mu0)
        val_w2 = float(w2_batch(nu0hat, nu0).item())
    print(f"epoch {epoch:03d}: train_loss={loss_val:.6f} | val_W2(mu0→nu0)={val_w2:.6f}")

    if val_w2 < best_val:
        best_val   = val_w2
        best_epoch = epoch
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "val_w2": best_val,
            "h_kernel": h_kernel,
            "BLUR": BLUR,
            "top_p": top_p,
            "keep_count": int(len(keep_idx)),
        }, best_model_path)
        disp_best = to_display_img(nu0hat, normalize_mass=True).detach().cpu()
        save_image(disp_best, best_img_path, nrow=1)

    if (epoch % save_every) == 0 or epoch == 1:
        with torch.no_grad():
            pred = model(mu0)
        disp = to_display_img(pred, normalize_mass=True).detach().cpu()
        save_image(disp, os.path.join(results_dir, f"pushforward_epoch_{epoch}.png"), nrow=1)

model.eval()
with torch.no_grad():
    nu0hat_final = model(mu0)
disp_final = to_display_img(nu0hat_final, normalize_mass=True).detach().cpu()
save_image(disp_final, os.path.join(results_dir, "pred_mu0_to_nu0hat_final.png"), nrow=1)

plt.figure()
plt.plot(loss_history, label="epoch_mean")
plt.plot(ema_history, label="EMA(0.9)")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curve.png")); plt.close()

if os.path.exists(best_model_path):
    chkpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(chkpt["model"])
    model.eval()
    with torch.no_grad():
        nu0hat_best = model(mu0)
    disp_best = to_display_img(nu0hat_best, normalize_mass=True).detach().cpu()
    save_image(disp_best, best_img_path, nrow=1)

print(
    f"Saved images in {results_dir}:\n"
    f"  mu0_{pair}.png\n  nu0_{pair}.png\n  pred_mu0_to_nu0hat_final.png\n"
    f"  pred_mu0_to_nu0hat_best.png (if improved)\n  pushforward_epoch_*.png (periodic)\n  loss_curve.png\n"
    f"Model checkpoint: {best_model_path} (best epoch {best_epoch}, val_W2={best_val:.6f})"
)
