from __future__ import annotations
import numpy as np

def make_modified_game(A: np.ndarray, B: np.ndarray, att_row: float, att_col: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Paper Eq.(1) in matrix form:
      Row utility: A' = A + att_row * B
      Col utility: B' = B + att_col * A
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return A + float(att_row) * B, B + float(att_col) * A
