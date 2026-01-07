from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

import nashpy as nash
from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import solve_robust, safe_probvec


# ---------- cooperation function (paper Eq. 3) ----------
def cooperation(att: float, bel: float) -> float:
    return (att + bel) / (np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0))


# ---------- compute expected observed probability ----------
def compute_fig2_curve(
    coop_grid,
    n_games=2000,
    n_actions=16,
    seed=0,
):
    rng = np.random.default_rng(seed)

    y = np.zeros_like(coop_grid, dtype=float)
    counts = np.zeros_like(coop_grid, dtype=int)

    for idx, k_target in enumerate(coop_grid):
        probs = []

        # Construct (att, bel) pairs that yield this cooperation level
        # Simple symmetric choice: att = bel
        # Then k = 2a / (a^2 + 1)
        # Solve numerically by scanning a
        a_vals = np.linspace(-1, 1, 2001)
        k_vals = (2 * a_vals) / (a_vals * a_vals + 1)
        a = a_vals[np.argmin(np.abs(k_vals - k_target))]

        att = float(a)
        bel = float(a)

        for _ in range(n_games):
            A, B = random_bimatrix_game(n_actions, rng=rng)

            # Modified game (opponent perspective only matters)
            A2, B2 = make_modified_game(A, B, att_row=bel, att_col=att)

            # Random Nash selection (as in paper)
            label = int(rng.integers(0, 2 * n_actions))
            res = solve_robust(nash.Game(A2, B2), label)
            if res is None:
                continue

            _, sigma = res
            sigma = safe_probvec(sigma, n_actions)

            # Sample move from sigma
            m = rng.choice(n_actions, p=sigma)
            probs.append(sigma[m])

        if probs:
            y[idx] = float(np.mean(probs))
            counts[idx] = len(probs)

    return y, counts


# ---------- plotting ----------
def plot_fig2(x, y, save_pdf=None):
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(x, y, "b-", linewidth=0.8)

    ax.set_title("Expected Probability of Observed Move Given Cooperation Level")
    ax.set_xlabel("Cooperation Level")
    ax.set_ylabel("Expected Observed Probability")

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.linspace(-1, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))

    ax.grid(False)
    plt.tight_layout()

    if save_pdf is not None:
        import os
        os.makedirs(os.path.dirname(save_pdf), exist_ok=True)
        fig.savefig(save_pdf, format="pdf", bbox_inches="tight")
        print(f"[INFO] saved {save_pdf}")

    plt.show()


# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--points", type=int, default=41)
    ap.add_argument("--actions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdf", type=str, default=None)
    args = ap.parse_args()

    coop_grid = np.linspace(-1, 1, args.points)

    y, counts = compute_fig2_curve(
        coop_grid,
        n_games=args.games,
        n_actions=args.actions,
        seed=args.seed,
    )

    print("[INFO] mean samples per point:", counts.mean())
    print("[INFO] min samples per point:", counts.min())

    plot_fig2(coop_grid, y, save_pdf=args.pdf)
