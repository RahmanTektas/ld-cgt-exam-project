# src/game.py
from __future__ import annotations
import numpy as np

def random_bimatrix_game(num_actions: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2-player normal-form (bi-matrix) game with payoffs in [0, 1].
    Returns:
      A: (num_actions, num_actions) payoffs for player 1
      B: (num_actions, num_actions) payoffs for player 2
    """
    A = rng.random((num_actions, num_actions))
    B = rng.random((num_actions, num_actions))
    return A, B
