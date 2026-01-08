from __future__ import annotations

import argparse
import os
import numpy as np
import nashpy as nash

from game import random_bimatrix_game
from modified_game import make_modified_game
from nash import solve_robust, safe_probvec
from tqdm import tqdm


def cooperation(att: float, bel: float) -> float:
    denom = np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0)
    return float((att + bel) / denom)


def bin_index(x: float, edges: np.ndarray) -> int:
    idx = np.searchsorted(edges, x, side="right") - 1
    return int(np.clip(idx, 0, len(edges) - 2))


def build_lookup_table(
    n_actions: int,
    n_samples_per_kl: int,
    prob_edges: np.ndarray,
    coop_edges: np.ndarray,
    error_levels: np.ndarray,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    J = len(prob_edges) - 1
    K = len(coop_edges) - 1
    L = len(error_levels)

    counts = np.zeros((J, K, L), dtype=np.int64)
    used = np.zeros((K, L), dtype=np.int64)   # how many valid samples contributed

    # We iterate over error levels, and for each, generate many synthetic (true, est) pairs
    for l, err in tqdm(
        list(enumerate(error_levels)),
        total=len(error_levels),
        desc="Building lookup (error levels)",
    ):
        for _ in tqdm(range(n_samples_per_kl * K), leave=False, desc=f"err={err:.3f}"):
        

            # Sample "true" opponent params
            att_true = float(rng.uniform(-1.0, 1.0))
            bel_true = float(rng.uniform(-1.0, 1.0))

            # Sample "estimated" params with Gaussian noise of scale err
            att_est = float(np.clip(att_true + rng.normal(0.0, err), -1.0, 1.0))
            bel_est = float(np.clip(bel_true + rng.normal(0.0, err), -1.0, 1.0))

            # Cooperation bin (based on estimated params)
            coop = float(np.clip(cooperation(att_est, bel_est), -1.0, 1.0))
            k = bin_index(coop, coop_edges)

            # Random game
            A, B = random_bimatrix_game(n_actions, rng=rng)

            # Build modified game using estimated opponent params
            A2, B2 = make_modified_game(A, B, att_row=bel_est, att_col=att_est)

            # Solve (random dropped label like in your runs)
            label = int(rng.integers(0, 2 * n_actions))
            res = solve_robust(nash.Game(A2, B2), label)
            if res is None:
                continue

            _, sigma_col = res
            sigma_col = safe_probvec(sigma_col, n_actions)

            # Sample an observed move and record its predicted probability
            m = int(rng.choice(n_actions, p=sigma_col))
            p_obs = float(sigma_col[m])
            j = bin_index(p_obs, prob_edges)

            counts[j, k, l] += 1
            used[k, l] += 1

    # Convert counts into conditional probability T[j,k,l] over j
    T = np.zeros_like(counts, dtype=float)
    for k in range(K):
        for l in range(L):
            s = counts[:, k, l].sum()
            if s > 0:
                T[:, k, l] = counts[:, k, l] / s
            else:
                # fallback if no samples: uniform over j
                T[:, k, l] = 1.0 / J

    return T, counts, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--J", type=int, default=20, help="number of probability bins")
    ap.add_argument("--K", type=int, default=20, help="number of cooperation bins")

    ap.add_argument("--samples", type=int, default=2000, help="rough samples per (K,L) bucket")
    ap.add_argument("--out", type=str, default="table.npz")

    ap.add_argument("--error_levels", type=str, default="0.05,0.1,0.2,0.4,0.8,1.2",
                    help="comma-separated error levels")
    args = ap.parse_args()

    error_levels = np.array([float(x) for x in args.error_levels.split(",")], dtype=float)

    # bins: p_obs in [0,1], coop in [-1,1]
    prob_edges = np.linspace(0.0, 1.0, args.J + 1)
    coop_edges = np.linspace(-1.0, 1.0, args.K + 1)

    T, counts, used = build_lookup_table(
        n_actions=args.actions,
        n_samples_per_kl=args.samples,
        prob_edges=prob_edges,
        coop_edges=coop_edges,
        error_levels=error_levels,
        seed=args.seed,
    )

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(
        args.out,
        T=T,
        prob_edges=prob_edges,
        coop_edges=coop_edges,
        error_levels=error_levels,
        counts=counts,
        used=used,
        meta=dict(
            actions=args.actions,
            seed=args.seed,
            J=args.J,
            K=args.K,
            samples=args.samples,
            error_levels=error_levels.tolist(),
        )
    )

    print("[INFO] saved:", args.out)
    print("[INFO] T shape:", T.shape)
    print("[INFO] used min/mean/max:", int(used.min()), float(used.mean()), int(used.max()))


if __name__ == "__main__":
    main()
