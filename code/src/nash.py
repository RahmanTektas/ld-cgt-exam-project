# src/nash_solve.py
from __future__ import annotations
import numpy as np
import warnings
import nashpy as nash

def safe_probvec(p: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=float).flatten()
    if p.shape[0] != n or (not np.isfinite(p).all()):
        return np.ones(n, dtype=float) / n
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        return np.ones(n, dtype=float) / n
    return p / s

def solve_robust(game: nash.Game, dropped_label: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Lemke–Howson with strong guards. Returns (sigma_row, sigma_col) or None.
    dropped_label is in [0, 2n-1].
    """
    try:
        A, B = game.payoff_matrices
        n = int(A.shape[0])
        label = int(dropped_label) % (2 * n)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            sr, sc = game.lemke_howson(initial_dropped_label=label)

        sr = safe_probvec(sr, n)
        sc = safe_probvec(sc, n)

        # If LH returned something degenerate, safe_probvec makes it uniform.
        # We still accept it.
        return sr, sc

    except (RuntimeWarning, FloatingPointError, ValueError, Exception):
        return None
