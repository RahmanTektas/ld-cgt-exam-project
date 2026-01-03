from __future__ import annotations
import numpy as np

def modified_game(A: np.ndarray, B: np.ndarray, att1: float, att2: float):
    """
    Player payoffs modified by attitudes:
      A' = A + att1 * B
      B' = B + att2 * A
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return A + att1 * B, B + att2 * A
