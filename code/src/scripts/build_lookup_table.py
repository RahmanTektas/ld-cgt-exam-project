import numpy as np
from tqdm import tqdm
from typing import Tuple

from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import one_nash_equilibrium


# --- Small helpers ------------------------------------------------------------

def cooperation(att: float, bel: float) -> float:
    """Cooperation score from the paper (bounded roughly in [-1, 1])."""
    denom = np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0)
    return float((att + bel) / denom)


def bin_index(x: float, edges: np.ndarray) -> int:
    """Return the bin index for x given bin edges."""
    idx = np.searchsorted(edges, x, side="right") - 1
    return int(np.clip(idx, 0, len(edges) - 2))


def safe_probvec(p: np.ndarray, n: int) -> np.ndarray:
    """Ensure p is a valid probability vector of length n; fallback to uniform."""
    p = np.asarray(p, dtype=float)
    if (not np.isfinite(p).all()) or p.sum() <= 0.0:
        return np.ones(n, dtype=float) / n
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return (p / s) if s > 0.0 else (np.ones(n, dtype=float) / n)


def sample_estimate_at_error(
    rng: np.random.Generator, att_true: float, bel_true: float, error: float
) -> Tuple[float, float]:
    """
    Create an estimated (att, bel) at a controlled distance 'error' from the true values.
    We move in a random direction by 'error' and clamp into [-1, 1].
    """
    direction = rng.normal(size=2)
    direction /= (np.linalg.norm(direction) + 1e-12)

    att_hat = float(np.clip(att_true + error * direction[0], -1.0, 1.0))
    bel_hat = float(np.clip(bel_true + error * direction[1], -1.0, 1.0))
    return att_hat, bel_hat


# --- Main builder -------------------------------------------------------------

def build_lookup_table(
    out_path: str = "lookup_table.npz",
    seed: int = 0,
    n_actions: int = 8,
    n_samples: int = 2000,
    J: int = 41,  # probability bins
    K: int = 41,  # cooperation bins
    error_levels: np.ndarray | None = None,  # L error levels
) -> None:
    """
    Build lookup table T[j,k,l] used to estimate the current error online.

    T[j,k,l] approximates:
        P(probability_bin = j | cooperation_bin = k, error_level = l)

    Output file contains: T, prob_edges, coop_edges, error_levels.
    """
    rng = np.random.default_rng(seed)

    if error_levels is None:
        error_levels = np.linspace(0.0, 1.0, 5)
    error_levels = np.asarray(error_levels, dtype=float)
    L = len(error_levels)

    prob_edges = np.linspace(0.0, 1.0, J + 1)
    coop_edges = np.linspace(-1.0, 1.0, K + 1)

    # counts[j,k,l] will be normalized into T[j,k,l]
    counts = np.zeros((J, K, L), dtype=np.float64)

    for _ in tqdm(range(n_samples), desc="Building lookup table"):
        # 1) Sample a random game
        A, B = random_bimatrix_game(n_actions, rng=rng)

        # 2) Sample the opponent's true parameters
        att_true = float(np.clip(rng.normal(0.0, 1.0), -1.0, 1.0))
        bel_true = float(np.clip(rng.normal(0.0, 1.0), -1.0, 1.0))
        nash_true = int(rng.integers(0, 2 * n_actions))

        # 3) Opponent chooses an action using its true model
        A_true, B_true = make_modified_game(A, B, att_row=bel_true, att_col=att_true)
        _, sigma_col_true = one_nash_equilibrium(A_true, B_true, nash_true)
        sigma_col_true = safe_probvec(sigma_col_true, n_actions)

        observed_move = int(rng.choice(n_actions, p=sigma_col_true))

        # 4) For each error level, create an estimate and record the probability bin
        for l, err in enumerate(error_levels):
            att_hat, bel_hat = sample_estimate_at_error(rng, att_true, bel_true, float(err))

            # Simple choice: keep the same Nash-selection label
            nash_hat = nash_true

            A_hat, B_hat = make_modified_game(A, B, att_row=bel_hat, att_col=att_hat)
            _, sigma_col_hat = one_nash_equilibrium(A_hat, B_hat, nash_hat)
            sigma_col_hat = safe_probvec(sigma_col_hat, n_actions)

            p_obs = float(sigma_col_hat[observed_move])  # predicted prob of the observed move

            j = bin_index(p_obs, prob_edges)
            k = bin_index(float(np.clip(cooperation(att_hat, bel_hat), -1.0, 1.0)), coop_edges)

            counts[j, k, l] += 1.0

    # 5) Normalize counts over j to get T[j,k,l] = P(j | k,l)
    T = counts.copy()
    for k in range(K):
        for l in range(L):
            s = T[:, k, l].sum()
            if s > 0.0:
                T[:, k, l] /= s
            else:
                T[:, k, l] = 1.0 / J

    np.savez_compressed(
        out_path,
        T=T,
        prob_edges=prob_edges,
        coop_edges=coop_edges,
        error_levels=error_levels,
        meta=dict(seed=seed, n_actions=n_actions, n_samples=n_samples, J=J, K=K, L=L),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    build_lookup_table()

