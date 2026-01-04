from __future__ import annotations
import numpy as np
import nashpy as nash
import warnings


def solve_robust(game, opponent_method_prediction):
    """
    Attempts to solve the game using Lemke-Howson.
    Returns (row_strategy, col_strategy) or None on failure.
    """
    try:
        with warnings.catch_warnings():
            # Treat numerical warnings as errors so we can catch them
            warnings.filterwarnings('error') 
            
            # 1. Get the equilibrium (In your version, this is a tuple of 2 arrays)
            eq = game.lemke_howson(initial_dropped_label=opponent_method_prediction)
            
            # 2. Check for NaNs
            if np.isnan(eq[0]).any() or np.isnan(eq[1]).any():
                return None
                
            return eq # This is (array_of_16, array_of_16)
                
    except (RuntimeWarning, ValueError, Exception):
        # Catches 'invalid value in divide' or any other solver crash
        return None