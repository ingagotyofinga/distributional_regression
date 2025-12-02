# loss.py
import torch
from geomloss import SamplesLoss

_sinkhorn_for_nn = SamplesLoss("sinkhorn", p=2, blur=0.1, debias=True, scaling=0.5, backend="auto")

def sinkhorn_cost(x, y, a=None, b=None):
    """
    GeomLoss calling convention:
      - unweighted:  L(x, y)
      - weighted:    L(a, x, b, y)   (weights first!)
    """
    if (a is None) and (b is None):
        return _sinkhorn_for_nn(x, y)
    # coerce to (N,) and match dtype/device
    a = torch.as_tensor(a, device=x.device, dtype=x.dtype).reshape(-1)
    b = torch.as_tensor(b, device=y.device, dtype=y.dtype).reshape(-1)
    return _sinkhorn_for_nn(a, x, b, y)

def wasserstein2_loss(T_mu, nu, *, a=None, b=None):
    return sinkhorn_cost(T_mu, nu, a, b)

def gaussian_kernel(mu, mu_0, bw=1.0, *, a=None, a0=None):
    with torch.no_grad():
        w2 = sinkhorn_cost(mu, mu_0, a, a0).double()
        h2 = torch.as_tensor(bw, dtype=torch.float64, device=w2.device) ** 2
        z  = (-w2 / h2).clamp(min=-50.0)
        k  = torch.exp(z).to(torch.float32)
    return k

def kernel_weighted_loss(
    mu, nu, mu_0, *, mode, pred=None, alpha=None, B=None,
    bw=0.5, return_normalizer=False,
    regime="gmm", mu_w=None, nu_w=None, mu0_w=None,
):
    use_weights = (regime is not None) and (regime.lower() == "mnist")

    k = gaussian_kernel(
        mu, mu_0, bw,
        a  = mu_w  if use_weights and (mu_w  is not None) else None,
        a0 = mu0_w if use_weights and (mu0_w is not None) else None,
    )

    if mode == "affine":
        assert alpha is not None and B is not None
        y_hat = mu @ B.T + alpha
        a_weights = mu_w if use_weights else None
        b_weights = nu_w if use_weights else None
    elif mode == "residual":
        assert pred is not None
        y_hat = pred
        a_weights = mu_w if use_weights else None
        b_weights = nu_w if use_weights else None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cost = sinkhorn_cost(y_hat, nu, a_weights, b_weights)
    loss = k * cost
    return (loss, k) if return_normalizer else loss