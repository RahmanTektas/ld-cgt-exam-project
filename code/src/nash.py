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
def solve_robust(
    game: nash.Game,
    dropped_label: int,
    *,
    rng: np.random.Generator | None = None,
    eps_schedule: tuple[float, ...] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6),
    tries_per_eps: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Robust Lemke–Howson wrapper with small payoff perturbations.
    Returns (sigma_row, sigma_col) or None on failure.

    Strategy:
      - Try Lemke–Howson on the original game (eps=0).
      - If it fails, add tiny Gaussian noise to payoffs and retry with same label.
    """
    try:
        A0, B0 = game.payoff_matrices
        A0 = np.asarray(A0, dtype=float)
        B0 = np.asarray(B0, dtype=float)

        n = int(A0.shape[0])
        label = int(dropped_label) % (2 * n)

        if rng is None:
            rng = np.random.default_rng()

        # scale noise to payoff magnitude so eps is meaningful
        scale = float(max(1.0, np.max(np.abs(A0)), np.max(np.abs(B0))))

        for eps in eps_schedule:
            # Build perturbed game
            if eps == 0.0:
                A, B = A0, B0
            else:
                A = A0 + (eps * scale) * rng.normal(0.0, 1.0, size=A0.shape)
                B = B0 + (eps * scale) * rng.normal(0.0, 1.0, size=B0.shape)

            g = nash.Game(A, B)

            for _ in range(int(tries_per_eps)):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")  # turn RuntimeWarning into exceptions
                        with np.errstate(divide="raise", invalid="raise", over="raise"):
                            sr, sc = g.lemke_howson(initial_dropped_label=label)

                    sr = np.asarray(sr, dtype=float).flatten()
                    sc = np.asarray(sc, dtype=float).flatten()

                    if sr.shape[0] != n or sc.shape[0] != n:
                        continue
                    if (not np.isfinite(sr).all()) or (not np.isfinite(sc).all()):
                        continue

                    sr = safe_probvec(sr, n)
                    sc = safe_probvec(sc, n)
                    return sr, sc

                except (RuntimeWarning, FloatingPointError, ValueError, ZeroDivisionError):
                    # try next attempt / next eps
                    continue
                except Exception:
                    continue

        return None

    except Exception:
        return None
