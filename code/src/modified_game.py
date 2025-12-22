# src/modified_game.py
from __future__ import annotations
import numpy as np

def make_modified_game(A: np.ndarray, B: np.ndarray, att_row: float, att_col: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Row: A' = A + att_row * B
    Col: B' = B + att_col * A
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return A + att_row * B, B + att_col * A
