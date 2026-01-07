from __future__ import annotations
import argparse
import numpy as np
import nashpy as nash

from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import solve_robust, safe_probvec


def generate_lookup_table(n_samples: int = 500, n_actions: int = 16, seed: int = 42):
    """
    Builds lookup_table.npz containing:
      T[j,k,l] ≈ P(observed prob-bin=j | cooperation-bin=k, error-level=l)
    Bins:
      prob_edges: 6 edges => 5 bins
      coop_edges: 6 edges => 5 bins
    """
    rng = np.random.default_rng(seed)

    prob_edges = np.linspace(0.0, 1.0, 6)   # 5 bins
    coop_edges = np.linspace(-1.0, 1.0, 6)  # 5 bins
    error_levels = np.array([0.1, 0.5, 1.0, 1.5], dtype=float)

    J = len(prob_edges) - 1
    K = len(coop_edges) - 1
    L = len(error_levels)

    T = np.ones((J, K, L), dtype=float)

    def coop(att: float, bel: float) -> float:
        denom = (np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0))
        return float((att + bel) / denom)

    def bin_index(x: float, edges: np.ndarray) -> int:
        idx = np.searchsorted(edges, x, side="right") - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    for l_idx, err_mag in enumerate(error_levels):
        valid = 0
        attempts = 0
        max_attempts = n_samples * 50  # give LH enough chances

        while valid < n_samples and attempts < max_attempts:
            attempts += 1

            # True opponent params
            att_true = rng.uniform(-1, 1)
            bel_true = rng.uniform(-1, 1)
            nash_label = int(rng.integers(0, 2 * n_actions))

            # Estimated params (perturb true by err_mag in random direction)
            angle = rng.uniform(0, 2 * np.pi)
            att_est = float(np.clip(att_true + err_mag * np.cos(angle), -1, 1))
            bel_est = float(np.clip(bel_true + err_mag * np.sin(angle), -1, 1))

            # Game
            A, B = random_bimatrix_game(n_actions, rng)

            # Opponent acts using true params: att_row = bel_true, att_col = att_true
            A_true, B_true = make_modified_game(A, B, att_row=bel_true, att_col=att_true)
            sol_true = solve_robust(nash.Game(A_true, B_true), nash_label)
            if sol_true is None:
                continue
            _, sigma_true = sol_true
            sigma_true = safe_probvec(sigma_true, n_actions)

            move = int(rng.choice(n_actions, p=sigma_true))

            # We predict using estimated params: att_row = bel_est, att_col = att_est
            A_est_m, B_est_m = make_modified_game(A, B, att_row=bel_est, att_col=att_est)
            sol_est = solve_robust(nash.Game(A_est_m, B_est_m), nash_label)
            if sol_est is None:
                continue
            _, sigma_est = sol_est
            sigma_est = safe_probvec(sigma_est, n_actions)

            prob_assigned = float(sigma_est[move])

            j = bin_index(prob_assigned, prob_edges)
            k = bin_index(np.clip(coop(att_est, bel_est), -1, 1), coop_edges)

            T[j, k, l_idx] += 1.0
            valid += 1

    # normalize over j for each (k,l)
    sums = T.sum(axis=0, keepdims=True)
    T = T / np.where(sums > 0, sums, 1.0)

    np.savez(
        "lookup_table.npz",
        T=T,
        prob_edges=prob_edges,
        coop_edges=coop_edges,
        error_levels=error_levels,
    )
    print(f"Saved lookup_table.npz with shape T={T.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--n_actions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    generate_lookup_table(n_samples=args.samples, n_actions=args.n_actions, seed=args.seed)
