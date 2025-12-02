import argparse
import time
from pathlib import Path

import torch
import pandas as pd

from src.data_utils.pair_dataset import DistributionPairDataset
from src.data_utils.metadata import load_test_pair
from src.loss import sinkhorn_cost  # GeomLoss SamplesLoss("sinkhorn", p=2, blur=0.15, debias=True)

# --- Regimes (keeping only d=2 for now as in your script) ---
regimes = {
    "A": (2, 100, 1000),
    "B": (2, 100, 100),
    "C": (2, 100, 5000),
    "D": (2, 10, 1000),
    "E": (2, 1000, 1000),
    # "F": (5, 100, 1000),
    # "G": (5, 100, 100),
    # "H": (5, 100, 10),
    # "I": (5, 10, 100),
}

# ---------------- helpers (keep simple & robust) ----------------

def _subsample(x: torch.Tensor, m: int) -> torch.Tensor:
    """Uniformly subsample rows of x to at most m points (no grad)."""
    with torch.no_grad():
        n = x.shape[0]
        if n <= m:
            return x
        idx = torch.randperm(n, device=x.device)[:m]
        return x.index_select(0, idx)

@torch.no_grad()
def _gather_targets(dataset, n_expected: int, s_per_target: int, dtype=torch.float32, device=None):
    """
    Returns a list of subsampled training targets [ν_i], each (<= s_per_target, d).
    Tries dataset.targets first; falls back to dataset[j][1].
    """
    t0 = time.time()
    tlist = []
    got_attr = hasattr(dataset, "targets")
    if got_attr:
        src = dataset.targets
        print(f"    [gather] using dataset.targets (len={len(src)})")
    else:
        print(f"    [gather] dataset.targets not found; using indexing")
        src = [dataset[j][1] for j in range(n_expected)]

    count = 0
    for nu in src:
        nu = nu.to(dtype=dtype, device=device, non_blocking=True)
        nu = _subsample(nu, s_per_target)
        tlist.append(nu)
        count += 1
        if count % 20 == 0:
            print(f"      [gather] processed {count}/{n_expected}")

    print(f"    [gather] done in {time.time() - t0:.2f}s, collected {len(tlist)} targets")
    return tlist

def compute_barycenter_small_support(target_list, M=512, iters=150, lr=1e-1, print_every=25):
    """
    Build a small-support entropic barycenter by optimizing M support points Y:
      Y := argmin_Y (1/|T|) sum_i Sinkhorn(Y, ν_i)
    Uses your sinkhorn_cost (debiased, blur=0.15). Returns Y (M,d) [no grad].
    """
    assert len(target_list) > 0, "target_list is empty"
    d = target_list[0].shape[1]
    device = target_list[0].device
    dtype = target_list[0].dtype

    # init Y from a small pool of targets
    init_pool = torch.cat(target_list[: min(4, len(target_list))], dim=0)
    Y = _subsample(init_pool, M).clone().detach().to(device=device, dtype=dtype).requires_grad_(True)

    opt = torch.optim.Adam([Y], lr=lr)

    t0 = time.time()
    for it in range(1, iters + 1):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for nu in target_list:
            loss = loss + sinkhorn_cost(Y, nu)
        loss = loss / len(target_list)
        loss.backward()
        opt.step()

        if (it == 1) or (it % print_every == 0) or (it == iters):
            print(f"      [bar] iter {it:03d}/{iters}  loss={float(loss):.6f}")

    print(f"    [bar] finished in {time.time() - t0:.2f}s (M={M}, iters={iters})")
    return Y.detach()

@torch.no_grad()
def compute_ss_tot_for_seed(dataset, tgt_test, *, s_per_target=512, M=512, iters=150, lr=1e-1, k_eval=4096):
    """
    1) Subsample each training target to <= s_per_target points
    2) Compute small-support barycenter with M points
    3) Subsample test target to k_eval points
    4) Return SS_tot = Sinkhorn(tgt_eval, Y_bar)
    """
    device = tgt_test.device
    dtype = tgt_test.dtype

    print("    [seed] downsampling targets...")
    tlist = _gather_targets(dataset, n_expected=len(dataset), s_per_target=s_per_target, dtype=dtype, device=device)

    print(f"    [seed] computing barycenter (M={M}, iters={iters})...")
    Y_bar = compute_barycenter_small_support(tlist, M=M, iters=iters, lr=lr)

    print(f"    [seed] subsampling test target to k_eval={k_eval}...")
    tgt_eval = _subsample(tgt_test, k_eval).to(dtype=dtype, device=device)

    print("    [seed] computing SS_tot (Sinkhorn)...")
    ss_tot = float(sinkhorn_cost(tgt_eval, Y_bar))
    print(f"    [seed] SS_tot = {ss_tot:.6f}")
    return ss_tot

# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    # optional knobs with safe defaults
    p.add_argument("--s_per_target", type=int, default=512)
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--k_eval", type=int, default=4096)
    p.add_argument("--seeds", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    records = []

    # ---------- compute SS_tot for each regime / seed ----------
    for regime_id, (dim, n, k) in regimes.items():
        print(f"\n[regime {regime_id}] dim={dim} n={n} k={k}")
        src_test, tgt_test = load_test_pair(k, dim)  # fixed test pair
        tgt_test = tgt_test.to(device=device, dtype=torch.float32)

        for ss in range(args.seeds):
            print(f"  [seed {ss}] loading subset...")
            t_seed0 = time.time()
            dataset = DistributionPairDataset(
                path=args.input_file,
                data_kind="gmm",
                n_sub=n,
                k_sub=k,
                subset_seed=ss,
                verbose=False,
            )
            print(f"  [seed {ss}] subset loaded in {time.time() - t_seed0:.2f}s; len={len(dataset)}")

            try:
                ss_tot = compute_ss_tot_for_seed(
                    dataset, tgt_test,
                    s_per_target=args.s_per_target,
                    M=args.M,
                    iters=args.iters,
                    lr=args.lr,
                    k_eval=args.k_eval,
                )
            except RuntimeError as e:
                print(f"  [seed {ss}] ERROR during SS_tot: {e}")
                ss_tot = float("nan")

            records.append(
                dict(
                    regime_id=regime_id,
                    subset_seed=ss,
                    dim=dim,
                    n=n,
                    k=k,
                    ss_tot=ss_tot,
                )
            )

            # free per-seed refs
            del dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_tot = pd.DataFrame(records)
    out_tot = out_dir / "ss_tot_by_seed.csv"
    df_tot.to_csv(out_tot, index=False)
    print(f"\n[write] saved SS_tot per seed → {out_tot}")

    # ---------- gather existing logs and merge ----------
    print("\n[logs] loading experiment logs...")
    dfs = {}
    for regime_id, (dim, _, _) in regimes.items():
        csv_path = Path(f"../results/d{dim}/gmm_{regime_id}/experiment_log.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["regime_id"] = regime_id
            dfs[regime_id] = df
            print(f"  [logs] loaded {csv_path} (rows={len(df)})")
        else:
            print(f"  [warn] missing: {csv_path}")

    if len(dfs) == 0:
        print("[logs] no logs found; exiting after SS_tot dump.")
        return

    df_all = pd.concat(dfs.values(), ignore_index=True)
    print(f"[logs] concatenated logs shape = {df_all.shape}")

    # merge SS_tot and compute R_W^2
    print("[merge] merging SS_tot with logs...")
    if "subset_seed" not in df_all.columns:
        raise KeyError("experiment_log.csv must have a 'subset_seed' column to merge on.")
    df_all = df_all.merge(df_tot, on=["regime_id", "subset_seed"], how="left")

    if "test_err_best" not in df_all.columns:
        raise KeyError("experiment_log.csv must include 'test_err_best' (SS_res).")

    # r2_w = 1 - SS_res / SS_tot
    df_all["r2_w"] = 1.0 - (df_all["test_err_best"] / df_all["ss_tot"])
    print("[merge] sample rows with r2_w:")
    print(df_all[["regime_id", "subset_seed", "test_err_best", "ss_tot", "r2_w"]].head())

    # save combined dataframe
    out_path = out_dir / "combined_regime_results.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n✅ Saved merged results with R_W^2 → {out_path}")

if __name__ == "__main__":
    main()
