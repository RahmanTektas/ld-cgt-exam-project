from __future__ import annotations
import numpy as np
import warnings
import nashpy as nash

EPS = 1e-12

def safe_probvec(p: np.ndarray, n: int, eps: float = EPS) -> np.ndarray:
    """
    Make p a valid probability vector of length n.
    Adds an epsilon floor to avoid exact zeros (prevents particle collapse).
    """
    p = np.asarray(p, dtype=float).flatten()
    if p.shape[0] != n or (not np.isfinite(p).all()):
        return np.ones(n, dtype=float) / n

    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        return np.ones(n, dtype=float) / n

    p = p / s
    # epsilon-floor smoothing (keeps support everywhere)
    p = np.clip(p, eps, None)
    p = p / float(p.sum())
    return p

def solve_robust(game: nash.Game, dropped_label: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Robust Lemke–Howson wrapper.
    Returns (sigma_row, sigma_col) or None on failure.
    """
    try:
        n = int(game.payoff_matrices[0].shape[0])
        label = int(dropped_label) % (2 * n)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            sr, sc = game.lemke_howson(initial_dropped_label=label)

        sr = np.asarray(sr, dtype=float).flatten()
        sc = np.asarray(sc, dtype=float).flatten()

        if sr.shape[0] != n or sc.shape[0] != n:
            return None
        if (not np.isfinite(sr).all()) or (not np.isfinite(sc).all()):
            return None

        sr = safe_probvec(sr, n)
        sc = safe_probvec(sc, n)
        return sr, sc

    except (RuntimeWarning, FloatingPointError, ValueError, Exception):
        return None
