# src/data/dataset_pairs.py

from __future__ import annotations
from typing import Optional, Literal, Sequence, Tuple
import os
import torch
from torch.utils.data import Dataset

DataKind = Literal["auto", "mnist", "gmm"]

class DistributionPairDataset(Dataset):
    """
    Supports:
      • GMM / sampled MNIST:  blob["mu"], blob["nu"]  (Tensors, shapes (N,K,2))
      • Weighted MNIST:       blob["mu_x"], "mu_w", "nu_x", "nu_w" (lists of Tensors)
    __getitem__ returns:
      • (mu, nu)                   for sampled/GMM
      • (mu_x, mu_w, nu_x, nu_w)   for weighted MNIST
    """
    def __init__(
        self,
        path: str,
        data_kind: DataKind = "auto",
        drop_intensity_for_mnist: bool = True,
        dtype: torch.dtype = torch.float32,
        # NEW: optional subsampling from a master file
        n_sub: Optional[int] = None,   # number of distributions to keep
        k_sub: Optional[int] = None,   # points per distribution to keep
        subset_seed: Optional[int] = None,  # controls which rows/cols we pick
        verbose: bool = True,          # quick one-line shape log
    ):
        blob = torch.load(path, map_location="cpu")
        meta = blob.get("meta", {})
        rep = meta.get("representation", None)  # "sampled" | "weighted" (if present)

        # --- Detect representation ---
        has_weighted = all(k in blob for k in ("mu_x", "mu_w", "nu_x", "nu_w"))
        has_sampled  = ("mu" in blob) and ("nu" in blob)
        if has_weighted and has_sampled:
            if rep == "sampled":
                has_weighted = False
            else:
                has_sampled = False

        self.meta = meta
        self.representation = "weighted" if has_weighted else "sampled"

        # --- Infer/force data kind (mnist/gmm) ---
        if data_kind == "auto":
            if isinstance(meta, dict) and meta.get("data_kind") in ("mnist", "gmm"):
                self.data_kind = meta["data_kind"]
            elif "mnist" in os.path.basename(path).lower():
                self.data_kind = "mnist"
            else:
                self.data_kind = "gmm"
        else:
            self.data_kind = data_kind

        self.weighted = (self.representation == "weighted")
        if self.weighted:
            # weighted path unchanged …
            def _fix_w(t, like):
                t = t.to(dtype).reshape(-1)
                s = t.sum()
                return (t / s) if s > 0 else torch.full((like.size(0),), 1.0 / like.size(0), dtype=dtype)
            self.mu_x = [t.to(dtype) for t in blob["mu_x"]]
            self.nu_x = [t.to(dtype) for t in blob["nu_x"]]
            self.mu_w = [_fix_w(w, x) for w, x in zip(blob["mu_w"], self.mu_x)]
            self.nu_w = [_fix_w(w, x) for w, x in zip(blob["nu_w"], self.nu_x)]
            self.N = len(self.mu_x)
            self.dim_ = int(self.mu_x[0].shape[1]) if self.mu_x else 2

        else:
            # --- sampled/GMM tensors ---
            mu = blob["mu"].to(dtype)
            nu = blob["nu"].to(dtype)

            # MNIST legacy fix: drop intensity channel if present
            if self.data_kind == "mnist" and drop_intensity_for_mnist and mu.size(-1) >= 3:
                mu = mu[..., :2]
                nu = nu[..., :2]

            if mu.shape != nu.shape:
                raise ValueError(f"mu/nu shape mismatch: {mu.shape} vs {nu.shape}")
            if mu.dim() != 3:
                raise ValueError(f"Expected (N,K,d) with d≥2; got {tuple(mu.shape)}")

            N_master, K_master, d = mu.shape
            if d < 2:
                raise ValueError(f"Expected last dim d≥2; got d={d}")


            # --- Optional subsampling ---
            if subset_seed is None:
                g = torch.Generator()
            else:
                g = torch.Generator().manual_seed(int(subset_seed))

            # pick distributions
            if n_sub is not None and n_sub < N_master:
                idxN = torch.randperm(N_master, generator=g)[:n_sub]
                mu = mu[idxN]
                nu = nu[idxN]
            # else keep all

            # pick points per distribution (same column indices for all dists for determinism)
            if k_sub is not None and k_sub != K_master:
                if k_sub < K_master:
                    idxK = torch.randperm(K_master, generator=g)[:k_sub]
                else:
                    # upsample with replacement if asked for more than available
                    idxK = torch.randint(0, K_master, (k_sub,), generator=g)
                mu = mu[:, idxK, :]
                nu = nu[:, idxK, :]

            self.mu, self.nu = mu.contiguous(), nu.contiguous()
            self.N = self.mu.size(0)
            self.dim_ = int(self.mu.size(-1))

            if verbose:
                print(f"[DistributionPairDataset] loaded {os.path.basename(path)} "
                      f"→ shape={tuple(self.mu.shape)} (N,K,d) kind={self.data_kind} subsample "
                      f"n_sub={n_sub} k_sub={k_sub} seed={subset_seed}")


    # ------------ Dataset API ------------
    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        if self.weighted:
            return self.mu_x[idx], self.mu_w[idx], self.nu_x[idx], self.nu_w[idx]
        else:
            return self.mu[idx], self.nu[idx]

    # ------------- Convenience -----------
    @property
    def is_weighted(self) -> bool:
        return self.weighted

    @property
    def dim(self) -> int:
        return self.dim_

    @property
    def shape(self) -> Tuple[int, int, int] | None:
        if self.weighted:
            return None
        return self.mu.shape  # (N,K,d)

    @property
    def n_pairs(self) -> int:
        return self.N

    def get_all_pairs(
        self,
        indices: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        detach: bool = False,
        numpy: bool = False,
        copy: bool = False,
    ):
        """
        Sampled/GMM: returns (mu, nu) tensors (maybe subset).
        Weighted: returns lists (mu_x_list, mu_w_list, nu_x_list, nu_w_list).
        """
        if indices is None:
            idxs = range(self.N)
        else:
            idxs = indices

        if not self.weighted:
            mu = self.mu if indices is None else self.mu[idxs]
            nu = self.nu if indices is None else self.nu[idxs]
            if copy:
                mu, nu = mu.clone(), nu.clone()
            if detach:
                mu, nu = mu.detach(), nu.detach()
            if device is not None:
                mu, nu = mu.to(device), nu.to(device)
            if numpy:
                mu, nu = mu.detach().cpu().numpy(), nu.detach().cpu().numpy()
            return mu, nu
        else:
            # build lists (ragged safe)
            mux = [self.mu_x[i].clone() if copy else self.mu_x[i] for i in idxs]
            muw = [self.mu_w[i].clone() if copy else self.mu_w[i] for i in idxs]
            nux = [self.nu_x[i].clone() if copy else self.nu_x[i] for i in idxs]
            nuw = [self.nu_w[i].clone() if copy else self.nu_w[i] for i in idxs]

            if detach:
                mux = [t.detach() for t in mux]; muw = [t.detach() for t in muw]
                nux = [t.detach() for t in nux]; nuw = [t.detach() for t in nuw]
            if device is not None:
                mux = [t.to(device) for t in mux]; muw = [t.to(device) for t in muw]
                nux = [t.to(device) for t in nux]; nuw = [t.to(device) for t in nuw]
            if numpy:
                mux = [t.detach().cpu().numpy() for t in mux]; muw = [t.detach().cpu().numpy() for t in muw]
                nux = [t.detach().cpu().numpy() for t in nux]; nuw = [t.detach().cpu().numpy() for t in nuw]
            return mux, muw, nux, nuw

    def to(self, device: torch.device):
        """In-place device move for sampled/GMM. Weighted uses per-item tensors—keep them on CPU or move on access."""
        if not self.weighted:
            self.mu = self.mu.to(device)
            self.nu = self.nu.to(device)
        return self


# Optional: a collate_fn for weighted batches if you ever use batch_size>1
def collate_pair(batch):
    """
    - For sampled/GMM: stacks to (B,k,2) as usual.
    - For weighted: returns lists for ragged support (you can keep batch_size=1 and ignore this).
    """
    if len(batch[0]) == 2:
        mu = torch.stack([b[0] for b in batch], dim=0)
        nu = torch.stack([b[1] for b in batch], dim=0)
        return mu, nu
    else:
        # keep ragged as lists
        mu_x = [b[0] for b in batch]
        mu_w = [b[1] for b in batch]
        nu_x = [b[2] for b in batch]
        nu_w = [b[3] for b in batch]
        return mu_x, mu_w, nu_x, nu_w
