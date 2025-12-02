import os, random, math
import numpy as np
import torch, torch.nn as nn
from pandas import concat
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
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
# Data: map 2 → 8 (adjust n as you like)
# -----------------------
train = MNIST(root=".", train=True, download=True, transform=ToTensor())

N_PER_CLASS = 500
pairs = [(0,5),(1,7),(2,8),(3,6),(4,9)]

def collect(ds, label, n=None):
    xs, ys = [], []
    for img, y in ds:
        if int(y) == label:
            xs.append(img)               # img is (1,28,28) if you used ToTensor()
            ys.append(label)
            if n is not None and len(xs) >= n:
                break
    X = torch.stack(xs, dim=0)          # (N,1,28,28)
    y = torch.tensor(ys, dtype=torch.long)  # (N,)
    return X, y

mu0s, nu0s = {}, {}
src_chunks, tgt_chunks = [], []
src_label_chunks, tgt_label_chunks = [], []
for s, t in pairs:
    src_all,src_all_labels = collect(train, s, N_PER_CLASS)          # CPU
    tgt_all,tgt_all_labels = collect(train, t, N_PER_CLASS)          # CPU

    mu0s[(s,t)] = src_all[:1].to(device)              # hold out on device
    nu0s[(s,t)] = tgt_all[:1].to(device)

    src_chunks.append(src_all[1:])                    # training pool (CPU for now)
    tgt_chunks.append(tgt_all[1:])
    src_label_chunks.append(src_all_labels[1:])
    tgt_label_chunks.append(tgt_all_labels[1:])

# full pool (CPU)
src = torch.cat(src_chunks, dim=0)                    # (5*(N-1),1,28,28)
tgt = torch.cat(tgt_chunks, dim=0)
src_labels = torch.cat(src_label_chunks, dim=0)
tgt_labels = torch.cat(tgt_label_chunks, dim=0)

assert src.shape[0] == src_labels.shape[0]
assert tgt.shape[0] == tgt_labels.shape[0]

M = src.shape[0]

# move full pool to device so kernel + model are consistent
src = src.to(device)
tgt = tgt.to(device)

# dataset with CPU indices (keep simple & consistent)
indices = torch.arange(M)                             # CPU
dataset = TensorDataset(src.cpu(), tgt.cpu(), indices)  # DataLoader expects CPU tensors
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# choose the pair you want to train manually (here: 2→8)
pair = (2,8)
mu0 = mu0s[pair]   # (1,1,28,28) on device
nu0 = nu0s[pair]

#------------------------
# Diagnostic
#------------------------

@torch.no_grad()
def kernel_neighbor_report(
    mu0,                   # (1,1,28,28)
    src,                   # (N,1,28,28)
    src_labels,            # (N,) long
    *,
    h=0.0075,              # your bandwidth / blur
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
    if kernel_fn is not None:
        w = kernel_fn(mu0, src, h=h)             # (N,)
    else:
        # Fallback: use your project's function name if it’s in scope:
        # from your_code import kernel_weights
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

h_kernel = 0.0019
lr = 1e-3
n_epochs = 200
base = 32

report = kernel_neighbor_report(
    mu0, src, src_labels, h=h_kernel, topk=32, tau=0.01, title="kernel@mu0"
)

run_name = f"{pair}_FULL_epochs{n_epochs}_h{h_kernel:.5f}_bl{BLUR:.3f}_b{base}_lr{lr:.0e}"
results_dir = os.path.join("../results/mnist",run_name)
os.makedirs(results_dir, exist_ok=True)

save_image(mu0, os.path.join(results_dir, f"mu0_{pair}.png"))
save_image(nu0, os.path.join(results_dir, f"nu0_{pair}.png"))

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
    print(f"[kernel@init] ESS={ess:.1f}/{M}  sum={W:.4f}  count(w>0.1)={cnt01}  count(w>0.05)={cnt005}")

    epoch_num = 0.0

    for (x_i, y_i, idx_i) in loader:
        # each is shape (1,1,28,28); idx_i is shape (1,)
        w_i = all_w[idx_i.item()]          # scalar tensor on device
        yhat_i = model(x_i)

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

    # Simple validation on (μ0 → ν0)
    model.eval()
    with torch.no_grad():
        nu0hat = model(mu0)
        val_w2 = float(w2_batch_nograd(nu0hat, nu0)[0].cpu())

    # ---- Update best and save preview/model ----
    if val_w2 < best_val:
        best_val = val_w2
        best_epoch = epoch
        # save weights
        torch.save({"epoch": epoch, "model": model.state_dict()}, best_model_path)
        # save the corresponding preview image
        disp_best = to_display_img(nu0hat, normalize_mass=True).detach().cpu()
        save_image(disp_best, best_img_path, nrow=1)
        print(f"[ckpt] saved best @ epoch {epoch}  val_W2={val_w2:.6f}")

    print(f"epoch {epoch:03d}: loss_mean={loss_epoch:.6f} | ema={ema:.6f} | val_W2(μ0→ν0)={val_w2:.6f}")

print(f"[done] best epoch={best_epoch}  best val_W2={best_val:.6f}  saved to {best_model_path}")

# -----------------------
# Save preview + loss curve (final weights + curve)
# -----------------------
model.eval()
with torch.no_grad():
    nu0hat = model(mu0)
disp = to_display_img(nu0hat, normalize_mass=True).detach().cpu()
save_image(disp, os.path.join(results_dir, "pred_mu0_to_nu0hat_final.png"), nrow=1)

plt.figure()
plt.plot(loss_history, label="epoch_mean")
plt.plot(ema_history, label="EMA(0.9)")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curve.png"))
plt.close()

# (Optional) also reload the best weights and resave the preview to be sure:
if os.path.exists(best_model_path):
    chkpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(chkpt["model"])
    with torch.no_grad():
        nu0hat_best = model(mu0)
    disp_best = to_display_img(nu0hat_best, normalize_mass=True).detach().cpu()
    save_image(disp_best, best_img_path, nrow=1)
