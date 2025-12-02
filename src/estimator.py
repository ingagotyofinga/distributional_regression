import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

__all__ = [
    "DistributionToTransformedSamplesNet",
    "DeepSetUNetTranslator",
    "build_estimator",
]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _zero_last_linear(seq: nn.Sequential) -> None:
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()
            break

def _init_softplus_bias_to(seq: nn.Sequential, target_value: float = 1.0) -> None:
    """Initialize the *last* Linear in seq so Softplus(linear(...)) ≈ target_value."""
    # Softplus^{-1}(y) = log(exp(y) - 1)
    bias_val = math.log(math.expm1(target_value))
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    m.bias.fill_(bias_val)
            break
# -----------------------------------------------------------------------------
# ORIGINAL ESTIMATOR (kept as-is)
# -----------------------------------------------------------------------------
class DistributionToTransformedSamplesNet(nn.Module):
    """
    mode='affine'   : y_hat = x @ B.T + alpha
    mode='residual' :
        - residual_head='free' : y_hat = x + Δ(x, context)     (original)
        - residual_head='mono' : y_hat = x0 + cumsum(s(x,ctx)*Δx), s>=0  (1D monotone)

    Input
    -----
    x : (N, d) points (sorted by the coordinate to be monotone if residual_head='mono')
    w : optional (N,) nonnegative weights summing ~1 (None -> uniform)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, mode: str = "affine",
                 residual_head: str = "free"):
        super().__init__()
        assert mode in ("affine", "residual")
        assert residual_head in ("free", "mono")
        self.d, self.h = input_dim, hidden_dim
        self.mode = mode
        self.residual_head = residual_head

        # per-point encoder (d -> h)
        self.point_encoder = nn.Sequential(
            nn.Linear(self.d, self.h), nn.ReLU(),
            nn.Linear(self.h, self.h), nn.ReLU(),
        )
        self.ln = nn.LayerNorm(self.h)

        self.global_decoder = nn.Sequential(
            nn.Linear(self.h, self.h), nn.ReLU(),
            nn.Linear(self.h, self.d + self.d * self.d),
        )

        # Original free residual head (kept as-is)
        self.delta_head = nn.Sequential(
            nn.Linear(self.d + self.h, self.h), nn.ReLU(),
            nn.Linear(self.h, self.d),
        )
        _zero_last_linear(self.delta_head)  # start near identity for residual

        # NEW: monotone 1D increments head (only used if residual_head='mono')
        # Produces s_i >= 0 per point; y = x0 + cumsum(s * Δx)
        self.mono_inc_head = nn.Sequential(
            nn.Linear(self.d + self.h, self.h), nn.ReLU(),
            nn.Linear(self.h, 1), nn.Softplus()
        )
        # initialize so s ≈ 1 → y ≈ x
        _init_softplus_bias_to(self.mono_inc_head, target_value=1.0)

    def _context(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.point_encoder(x)  # (N,h)
        if w is None:
            z = h.mean(dim=0)
        else:
            w = w / (w.sum() + 1e-12)
            z = (h * w[:, None]).sum(dim=0)
        z = self.ln(z)
        return z

    def encode(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._context(x, w)

    # ---------- affine params ----------
    def predict_params(self, x: torch.Tensor, w: Optional[torch.Tensor] = None):
        ctx = self._context(x, w)                 # (h,)
        out = self.global_decoder(ctx)            # (d + d^2,)
        alpha = out[: self.d]                     # (d,)
        B = out[self.d :].view(self.d, self.d)    # (d,d)
        return alpha, B

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "affine":
            alpha, B = self.predict_params(x, w)
            return x @ B.T + alpha

        # residual mode
        if self.residual_head == "free":
            ctx = self._context(x, w)  # (h,)
            ctx_per_point = ctx.expand(x.size(0), -1)  # (N,h)
            feats = torch.cat([x, ctx_per_point], dim=-1)
            delta = self.delta_head(feats)  # (N,d)
            return x + delta

        # monotone 1D head
        # NOTE: requires d == 1 and x sorted ascending along age
        assert self.d == 1, "residual_head='mono' currently supports 1D inputs only."
        # compute positive increments conditioned on (x, context)
        ctx = self._context(x, w)               # (h,)
        ctx_per_point = ctx.expand(x.size(0), -1)  # (N,h)
        feats = torch.cat([x, ctx_per_point], dim=-1)  # (N, 1+h)
        s = self.mono_inc_head(feats).view(-1)  # (N,), s >= 0

        # Δx per step (supports non-unit grids)
        x1d = x.view(-1)
        dx = torch.empty_like(x1d)
        dx[0] = (x1d[1] - x1d[0]) if x1d.numel() > 1 else x1d.new_tensor(1.0)
        if x1d.numel() > 1:
            dx[1:] = x1d[1:] - x1d[:-1]

        y = x1d[0] + torch.cumsum(s * dx, dim=0)   # strictly increasing
        return y.view_as(x)  # (N,1)

# -----------------------------------------------------------------------------
# NEW: DeepSets "Set U-Net" backbone + translator (Option A)
# -----------------------------------------------------------------------------
class WeightedPool(nn.Module):
    """Weighted mean over points with numerical stability."""
    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (N, C); w: (N,)
        if w is None:
            return x.mean(dim=0)
        ws = w / (w.sum() + 1e-12)
        return (x * ws[:, None]).sum(dim=0)

class SetMLP(nn.Module):
    """Point-wise MLP with LayerNorm and ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, out_ch)
        self.lin2 = nn.Linear(out_ch, out_ch)
        self.ln1 = nn.LayerNorm(out_ch)
        self.ln2 = nn.LayerNorm(out_ch)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lin1.weight, nonlinearity="relu")
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity="relu")
        nn.init.zeros_(self.lin2.bias)
        nn.init.ones_(self.ln1.weight); nn.init.zeros_(self.ln1.bias)
        nn.init.ones_(self.ln2.weight); nn.init.zeros_(self.ln2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.ln1(self.lin1(x)))
        x = nn.functional.relu(self.ln2(self.lin2(x)))
        return x

class FiLMGate(nn.Module):
    """Global context → per-channel affine (Feature-wise Linear Modulation)."""
    def __init__(self, ch: int):
        super().__init__()
        self.gamma = nn.Linear(ch, ch)
        self.beta = nn.Linear(ch, ch)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, feats: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # feats: (N, C), ctx: (C,)
        g = self.gamma(ctx)  # (C,)
        b = self.beta(ctx)   # (C,)
        return feats * (1 + g)[None, :] + b[None, :]

class SetEncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = SetMLP(in_ch, out_ch)
        self.pool = WeightedPool()
        self.gate = FiLMGate(out_ch)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        z = self.pool(h, w)         # (C,)
        h = self.gate(h, z)
        return h, z

class SetDecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = SetMLP(in_ch + skip_ch, out_ch)
        self.pool = WeightedPool()
        self.gate = FiLMGate(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x, skip], dim=-1)
        h = self.block(h)
        z = self.pool(h, w)
        h = self.gate(h, z)
        return h, z

class DeepSetsUNet(nn.Module):
    """
    Permutation-invariant U-Net for point sets with optional weights.
    - Input: points x ∈ R^{N×d}, optional weights w ∈ R^N (nonnegative).
    - Output: per-point features in R^{N×H}, plus a global context vector in R^{H}.
    """
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Linear(d, hidden)
        self.enc1 = SetEncoderStage(hidden, hidden)
        self.enc2 = SetEncoderStage(hidden, hidden)
        self.bottleneck = SetEncoderStage(hidden, hidden)
        self.dec2 = SetDecoderStage(hidden, hidden, hidden)
        self.dec1 = SetDecoderStage(hidden, hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        nn.init.kaiming_normal_(self.embed.weight, nonlinearity="relu")
        nn.init.zeros_(self.embed.bias)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N,d)
        h0 = nn.functional.relu(self.embed(x))      # (N,H)
        e1, z1 = self.enc1(h0, w)                   # (N,H), (H,)
        e2, z2 = self.enc2(e1, w)                   # (N,H), (H,)
        b, zb = self.bottleneck(e2, w)              # (N,H), (H,)
        d2, _ = self.dec2(b, e2, w)                 # (N,H)
        d1, _ = self.dec1(d2, e1, w)                # (N,H)
        out = self.ln(d1)                           # (N,H)
        if w is None:
            z = out.mean(dim=0)
        else:
            ws = w / (w.sum() + 1e-12)
            z = (out * ws[:, None]).sum(dim=0)
        return out, z

class DeepSetUNetTranslator(nn.Module):
    """
    Drop-in replacement estimator using a DeepSets U-Net backbone.

    mode='affine'   : forward(x,w) returns x @ B^T + alpha  (B, alpha from global code)
    mode='residual' : forward(x,w) returns x + Δ(x, context)

    Input/Output: identical to DistributionToTransformedSamplesNet
    - x: (N, d)
    - w: optional (N,) weights
    - returns: (N, d)
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, mode: str = "residual"):
        super().__init__()
        assert mode in ("affine", "residual")
        self.d = input_dim
        self.h = hidden_dim
        self.mode = mode

        self.backbone = DeepSetsUNet(d=self.d, hidden=self.h)

        # Heads
        self.delta_head = nn.Sequential(
            nn.Linear(self.h + self.d, self.h), nn.ReLU(),
            nn.Linear(self.h, self.d),
        )
        _zero_last_linear(self.delta_head)  # start near identity

        self.param_head = nn.Sequential(
            nn.Linear(self.h, self.h), nn.ReLU(),
            nn.Linear(self.h, self.d + self.d * self.d),
        )
        # initialize the last linear mildly for stability
        last = list(self.param_head.modules())[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    # keep parity with old estimator API
    def encode(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats, z = self.backbone(x, w)
        return z

    def predict_params(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, z = self.backbone(x, w)  # feats: (N,H), z: (H,)
        out = self.param_head(z)
        alpha = out[: self.d]
        B = out[self.d :].view(self.d, self.d)
        return alpha, B

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "affine":
            alpha, B = self.predict_params(x, w)
            return x @ B.T + alpha
        # residual mode
        feats, z = self.backbone(x, w)              # (N,H), (H,)
        ctx = z.expand(x.size(0), -1)               # (N,H)
        delta = self.delta_head(torch.cat([x, ctx], dim=-1))  # (N,d)
        return x + delta

# -----------------------------------------------------------------------------
# Simple factory to keep train.py clean
# -----------------------------------------------------------------------------

def build_estimator(
    *,
    arch: str = "mlp",
    input_dim: int = 2,
    hidden_dim: int = 128,
    mode: str = "residual",
    residual_head="mono"
) -> nn.Module:
    """
    arch: "mlp" (original) or "set_unet" (DeepSets U-Net)
    mode: "affine" or "residual"
    """
    arch = arch.lower()
    if arch == "mlp":
        return DistributionToTransformedSamplesNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            mode=mode,
            residual_head=residual_head
        )
    if arch == "set_unet":
        return DeepSetUNetTranslator(input_dim=input_dim, hidden_dim=hidden_dim, mode=mode)
    raise ValueError(f"Unknown arch={arch!r}. Use 'mlp' or 'set_unet'.")
