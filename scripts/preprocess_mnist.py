import argparse
import os
import random
from collections import defaultdict
from typing import Tuple

import torch
from torchvision import datasets, transforms


# ---- Helpers --------------------------------------------------------------

def _pixel_grid_coords(H: int, W: int, coord_range: str = "unit") -> torch.Tensor:
    """
    Return (H*W, 2) coordinates for pixel centers.
    Order = (row, col) to match your existing convention.
    """
    rows = torch.arange(H, dtype=torch.float32)
    cols = torch.arange(W, dtype=torch.float32)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    r = (rr + 0.5) / H
    c = (cc + 0.5) / W
    if coord_range == "minus1_1":
        r = r * 2 - 1
        c = c * 2 - 1
    return torch.stack([r.reshape(-1), c.reshape(-1)], dim=1)  # (H*W, 2), (row, col)


@torch.no_grad()
def image_to_empirical_points(
    img: torch.Tensor,
    k: int,
    jitter: bool = True,
    eps: float = 1e-12,
    coord_range: str = "unit",
) -> torch.Tensor:
    """
    OLD behavior: sample k points from image intensities -> (k,2) point cloud.
    """
    if img.dim() == 3:
        img = img.squeeze(0)  # (H, W)
    H, W = img.shape

    a = img.clamp_min(0).reshape(-1)
    s = a.sum()
    if s <= eps:
        p = torch.full_like(a, 1.0 / a.numel())
    else:
        p = a / s

    idx = torch.multinomial(p, num_samples=k, replacement=True)  # (k,)
    rows = idx // W
    cols = idx % W

    if jitter:
        r = (rows.to(torch.float32) + torch.rand_like(rows, dtype=torch.float32)) / H
        c = (cols.to(torch.float32) + torch.rand_like(cols, dtype=torch.float32)) / W
    else:
        r = (rows.to(torch.float32) + 0.5) / H
        c = (cols.to(torch.float32) + 0.5) / W

    if coord_range == "minus1_1":
        r = r * 2 - 1
        c = c * 2 - 1

    pts = torch.stack([r, c], dim=1)  # (k, 2), order = (row, col)
    return pts


@torch.no_grad()
def image_to_weighted_measure(
    img: torch.Tensor,
    coord_range: str = "unit",
    thresh: float = 0.0,
    topk: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NEW behavior: return full weighted discrete measure (coords, weights).
      - coords: (M, 2) pixel centers (M=H*W unless sparsified)
      - weights: (M,) nonnegative, normalized to sum 1
    Optional sparsification via 'thresh' or 'topk'.
    """
    if img.dim() == 3:
        img = img.squeeze(0)  # (H, W)
    H, W = img.shape

    coords = _pixel_grid_coords(H, W, coord_range=coord_range)  # (H*W, 2)

    w = img.float().reshape(-1).clamp_min(0.0)  # raw intensities
    if w.max() > 1.0:
        w = w / 255.0

    # Optional sparsification
    keep = torch.ones_like(w, dtype=torch.bool)
    if thresh is not None and thresh > 0.0:
        keep &= (w > thresh)

    if topk is not None:
        # Keep largest 'topk' masses; combine with threshold if both set
        k = min(topk, int(keep.sum().item()) if keep.any() else w.numel())
        if k > 0:
            idx = torch.topk(w * keep, k).indices
            sel = torch.zeros_like(keep)
            sel[idx] = True
            keep = sel

    coords = coords[keep]
    w = w[keep]

    # If everything got dropped, fall back to uniform over full grid
    if w.numel() == 0 or float(w.sum()) == 0.0:
        coords = _pixel_grid_coords(H, W, coord_range=coord_range)
        w = torch.full((H * W,), 1.0 / (H * W), dtype=torch.float32)
        return coords, w

    w = w / (w.sum() + 1e-12)
    return coords, w


# ---- Build paired dataset ------------------------------------------------

def build_paired(
    split: str,
    mapping: dict[int, int],
    limit_per_source: int | None,
    seed: int,
):
    """
    Get MNIST dataset + label buckets + paired indices once.
    """
    random.seed(seed)

    ds = datasets.MNIST(
        root="./data/raw/MNIST",
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )

    # Bucket indices by label
    buckets = defaultdict(list)
    for idx, (_, y) in enumerate(ds):
        buckets[int(y)].append(idx)

    # Prepare index pairs (source -> random target in mapped class)
    pair_indices = []
    for src_label, tgt_label in mapping.items():
        src_pool = buckets[src_label]
        tgt_pool = buckets[tgt_label]
        if not src_pool or not tgt_pool:
            continue

        n_src = len(src_pool) if limit_per_source is None else min(len(src_pool), limit_per_source)
        for i in range(n_src):
            si = src_pool[i]
            tj = random.choice(tgt_pool)
            pair_indices.append((si, tj))

    if not pair_indices:
        raise RuntimeError("No pairs created. Check your mapping and dataset split.")

    return ds, pair_indices


def build_paired_sampled(
    split: str,
    k: int,
    mapping: dict[int, int],
    limit_per_source: int | None,
    seed: int,
    jitter: bool,
    coord_range: str,
):
    ds, pair_indices = build_paired(split, mapping, limit_per_source, seed)

    mu_list, nu_list = [], []
    for si, tj in pair_indices:
        img_s, _ = ds[si]  # (1,28,28)
        img_t, _ = ds[tj]
        mu = image_to_empirical_points(img_s, k=k, jitter=jitter, coord_range=coord_range)
        nu = image_to_empirical_points(img_t, k=k, jitter=jitter, coord_range=coord_range)
        mu_list.append(mu)
        nu_list.append(nu)

    mu_tensor = torch.stack(mu_list)  # (n_pairs, k, 2)
    nu_tensor = torch.stack(nu_list)  # (n_pairs, k, 2)
    return mu_tensor, nu_tensor


def build_paired_weighted(
    split: str,
    mapping: dict[int, int],
    limit_per_source: int | None,
    seed: int,
    coord_range: str,
    thresh: float,
    topk: int | None,
):
    ds, pair_indices = build_paired(split, mapping, limit_per_source, seed)

    mu_x, mu_w, nu_x, nu_w = [], [], [], []
    for si, tj in pair_indices:
        img_s, _ = ds[si]  # (1,28,28)
        img_t, _ = ds[tj]

        x_s, w_s = image_to_weighted_measure(img_s, coord_range=coord_range, thresh=thresh, topk=topk)
        x_t, w_t = image_to_weighted_measure(img_t, coord_range=coord_range, thresh=thresh, topk=topk)

        mu_x.append(x_s); mu_w.append(w_s)
        nu_x.append(x_t); nu_w.append(w_t)

    # Ragged support possible if using thresh/topk; stack into lists (torch.save handles lists of tensors)
    return mu_x, mu_w, nu_x, nu_w


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", choices=["sampled", "weighted"], default="sampled",
                        help="Representation: sampled point clouds (old) or weighted pixel measures (new).")
    parser.add_argument("--k", type=int, default=1000, help="samples per image (for --rep=sampled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_source", type=int, default=None,
                        help="cap #source images per source class")
    parser.add_argument("--jitter", action="store_true", help="jitter within pixel cells (sampled only)")
    parser.add_argument("--coord_range", choices=["unit", "minus1_1"], default="unit")
    parser.add_argument("--outdir", type=str, default="./data/generated/mnist")
    parser.add_argument("--suffix", type=str, default="", help="optional filename suffix like _weighted or _k100")
    # weighted options
    parser.add_argument("--thresh", type=float, default=0.0, help="drop pixels with intensity <= thresh (weighted only)")
    parser.add_argument("--topk", type=int, default=None, help="keep top-k most massive pixels (weighted only)")
    # Mapping like "2:8,4:6"
    parser.add_argument("--mapping", type=str, default="2:8,4:6")
    args = parser.parse_args()

    # Parse mapping string
    mapping = {}
    for pair in args.mapping.split(","):
        a, b = pair.split(":")
        mapping[int(a)] = int(b)

    os.makedirs(args.outdir, exist_ok=True)

    # Filenames
    suf = args.suffix if args.suffix else (f"_{args.rep}" if args.rep else "")
    train_path = os.path.join(args.outdir, f"mnist_train{suf}.pt")
    test_path  = os.path.join(args.outdir, f"mnist_test{suf}.pt")

    if args.rep == "sampled":
        # ---- Train ----
        mu_train, nu_train = build_paired_sampled(
            split="train",
            k=args.k,
            mapping=mapping,
            limit_per_source=args.limit_per_source,
            seed=args.seed,
            jitter=args.jitter,
            coord_range=args.coord_range,
        )
        torch.save(
            {
                "mu": mu_train,  # (n_pairs, k, 2)
                "nu": nu_train,  # (n_pairs, k, 2)
                "meta": {
                    "data_kind": "mnist",
                    "representation": "sampled",
                    "k": int(args.k),
                    "coord_range": args.coord_range,
                    "jitter": bool(args.jitter),
                },
            },
            train_path,
        )

        # ---- Test ----
        mu_test, nu_test = build_paired_sampled(
            split="test",
            k=args.k,
            mapping=mapping,
            limit_per_source=1,   # small sample
            seed=args.seed + 1,
            jitter=args.jitter,
            coord_range=args.coord_range,
        )
        torch.save(
            {
                "mu": mu_test,
                "nu": nu_test,
                "meta": {
                    "data_kind": "mnist",
                    "representation": "sampled",
                    "k": int(args.k),
                    "coord_range": args.coord_range,
                    "jitter": bool(args.jitter),
                },
            },
            test_path,
        )

        print(f"Saved sampled MNIST to {args.outdir}: "
              f"{os.path.basename(train_path)} (pairs={mu_train.shape[0]}), "
              f"{os.path.basename(test_path)} (pairs={mu_test.shape[0]})")

    else:  # args.rep == "weighted"
        # ---- Train ----
        mu_x_tr, mu_w_tr, nu_x_tr, nu_w_tr = build_paired_weighted(
            split="train",
            mapping=mapping,
            limit_per_source=args.limit_per_source,
            seed=args.seed,
            coord_range=args.coord_range,
            thresh=args.thresh,
            topk=args.topk,
        )
        torch.save(
            {
                # lists of tensors (ragged allowed if thresh/topk used)
                "mu_x": mu_x_tr, "mu_w": mu_w_tr,
                "nu_x": nu_x_tr, "nu_w": nu_w_tr,
                "meta": {
                    "data_kind": "mnist",
                    "representation": "weighted",
                    "coord_range": args.coord_range,
                    "thresh": float(args.thresh),
                    "topk": None if args.topk is None else int(args.topk),
                },
            },
            train_path,
        )

        # ---- Test ----
        mu_x_te, mu_w_te, nu_x_te, nu_w_te = build_paired_weighted(
            split="test",
            mapping=mapping,
            limit_per_source=1,
            seed=args.seed + 1,
            coord_range=args.coord_range,
            thresh=args.thresh,
            topk=args.topk,
        )
        torch.save(
            {
                "mu_x": mu_x_te, "mu_w": mu_w_te,
                "nu_x": nu_x_te, "nu_w": nu_w_te,
                "meta": {
                    "data_kind": "mnist",
                    "representation": "weighted",
                    "coord_range": args.coord_range,
                    "thresh": float(args.thresh),
                    "topk": None if args.topk is None else int(args.topk),
                },
            },
            test_path,
        )

        print(f"Saved weighted MNIST to {args.outdir}: "
              f"{os.path.basename(train_path)} (pairs={len(mu_x_tr)}), "
              f"{os.path.basename(test_path)} (pairs={len(mu_x_te)})")


if __name__ == "__main__":
    main()
