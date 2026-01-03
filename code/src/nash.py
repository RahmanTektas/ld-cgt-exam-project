from __future__ import annotations
import numpy as np
import nashpy as nash
import warnings


def one_nash_equilibrium(A: np.ndarray, B: np.ndarray, dropped_label: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    One mixed NE via Lemke–Howson. The 'method of picking NE' is modeled
    by the initial_dropped_label
    """
    game = nash.Game(A, B)

    # Nashpy expects label in [0, 2*n - 1]
    n = A.shape[0]
    label = int(dropped_label) % (2 * n)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
           category=RuntimeWarning,
        )
        sigma_row, sigma_col = game.lemke_howson(initial_dropped_label=label)

    sr = np.clip(np.asarray(sigma_row, float), 0.0, None)
    sc = np.clip(np.asarray(sigma_col, float), 0.0, None)

    if sr.sum() == 0 or sc.sum() == 0:
        u = np.ones(n) / n
        return u, u

    return sr / sr.sum(), sc / sc.sum()
