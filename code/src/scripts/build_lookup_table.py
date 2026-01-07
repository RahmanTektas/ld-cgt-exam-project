# tools/build_lookup.py
from __future__ import annotations
import numpy as np
import nashpy as nash
import argparse

from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import solve_robust, safe_probvec

def coop(att: float, bel: float) -> float:
    denom = (np.sqrt(att*att + 1.0) * np.sqrt(bel*bel + 1.0))
    return float((att + bel) / denom)

def bin_index(x: float, edges: np.ndarray) -> int:
    idx = np.searchsorted(edges, x, side="right") - 1
    return int(np.clip(idx, 0, len(edges) - 2))

def generate_lookup(n_samples=500, seed=42, out="lookup_table.npz"):
    n_actions = 16

    prob_edges = np.linspace(0.0, 1.0, 6)     # 5 bins
    coop_edges = np.linspace(-1.0, 1.0, 6)    # 5 bins
    error_levels = np.array([0.1, 0.5, 1.0, 1.5], dtype=float)

    J = len(prob_edges) - 1
    K = len(coop_edges) - 1
    L = len(error_levels)

    # counts
    T = np.ones((J, K, L), dtype=float)  # Laplace smoothing

    rng = np.random.default_rng(seed)

    for l_idx, err_mag in enumerate(error_levels):
        valid = 0
        attempts = 0

        while valid < n_samples and attempts < n_samples * 10:
            attempts += 1

            # True opponent parameters
            att_true = rng.uniform(-1, 1)
            bel_true = rng.uniform(-1, 1)

            # Estimated parameters at controlled "error radius"
            ang = rng.uniform(0, 2*np.pi)
            att_est = float(np.clip(att_true + err_mag*np.cos(ang), -1, 1))
            bel_est = float(np.clip(bel_true + err_mag*np.sin(ang), -1, 1))

            A, B = random_bimatrix_game(n_actions, rng)

            # Opponent uses reciprocation internally:
            # agent_used_att  = clip(att_true + r) is NOT needed here
            # opponent_used_att = clip(bel_true + r) depends on its belief
            r = 0.1
            att_opp_used_true = float(np.clip(bel_true + r, -1, 1))
            # For generating an action we need the NE of the modified game under "true used attitudes".
            # We also need some "our used attitude" value: assume symmetric reciprocation using opponent's att_true:
            att_agent_used_true = float(np.clip(att_true + r, -1, 1))

            label = int(rng.integers(0, 2*n_actions))

            A_true, B_true = make_modified_game(A, B, att_agent_used_true, att_opp_used_true)
            sol_true = solve_robust(nash.Game(A_true, B_true), label)
            if sol_true is None:
                continue
            _, sigma_col_true = sol_true
            sigma_col_true = safe_probvec(sigma_col_true, n_actions)

            move = int(rng.choice(n_actions, p=sigma_col_true))

            # Prediction using estimated params (same forward model)
            att_opp_used_est = float(np.clip(bel_est + r, -1, 1))
            att_agent_used_est = float(np.clip(att_est + r, -1, 1))

            A_est, B_est = make_modified_game(A, B, att_agent_used_est, att_opp_used_est)
            sol_est = solve_robust(nash.Game(A_est, B_est), label)
            if sol_est is None:
                continue
            _, sigma_col_est = sol_est
            sigma_col_est = safe_probvec(sigma_col_est, n_actions)

            p_assigned = float(sigma_col_est[move])

            j = bin_index(p_assigned, prob_edges)
            k = bin_index(np.clip(coop(att_est, bel_est), -1, 1), coop_edges)

            T[j, k, l_idx] += 1.0
            valid += 1

    # Normalize: for each (k,l), probabilities over j sum to 1
    denom = T.sum(axis=0, keepdims=True)
    T = T / np.where(denom > 0, denom, 1.0)

    np.savez(out, T=T, prob_edges=prob_edges, coop_edges=coop_edges, error_levels=error_levels)
    print(f"Saved {out} with shape T={T.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--out", type=str, default="lookup_table.npz")
    args = parser.parse_args()
    generate_lookup(n_samples=args.samples, out=args.out)
