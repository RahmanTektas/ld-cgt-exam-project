import numpy as np
import nashpy as nash
import warnings
import argparse

# Adjust these imports if your folder structure requires 'src.' prefix
from src.game import random_bimatrix_game
from src.modified_game import make_modified_game

# We define solve_robust locally to ensure it handles shapes correctly
# and avoids import signature mismatches.
def solve_robust(game, opponent_method_prediction):
    """
    Attempts to solve the game using Lemke-Howson.
    Returns (row_strategy, col_strategy) or None on failure.
    """
    try:
        with warnings.catch_warnings():
            # Treat numerical warnings as errors so we can catch them
            warnings.filterwarnings('error') 
            
            # Lemke-Howson requires an integer label to drop
            eq = game.lemke_howson(initial_dropped_label=int(opponent_method_prediction))
            
            row_s, col_s = eq
            
            # Check for NaNs
            if np.isnan(row_s).any() or np.isnan(col_s).any():
                return None
                
            return row_s, col_s
                
    except (RuntimeWarning, ValueError, Exception):
        return None

def generate_simple_table(n_samples=500):
    print(f"Generating simplified lookup table ({n_samples} samples/level)...")
    
    n_actions = 16
    
    # 1. Setup simple bins
    prob_bins = np.linspace(0, 1, 6)   # 5 bins
    coop_bins = np.linspace(-1, 1, 6)  # 5 bins
    error_levels = np.array([0.1, 0.5, 1.0, 1.5]) # 4 levels
    
    # Shape: [Prob, Coop, Error]
    T = np.ones((len(prob_bins)-1, len(coop_bins)-1, len(error_levels)))
    
    rng = np.random.default_rng(42)
    
    # 2. Simulation Loop
    for l_idx, err_mag in enumerate(error_levels):
        valid_samples = 0
        
        # We loop until we get n_samples valid data points for this level
        # or give up after too many tries (to prevent infinite loops)
        attempts = 0
        while valid_samples < n_samples and attempts < n_samples * 2:
            attempts += 1
            
            # Random "True" Opponent
            att = rng.uniform(-1, 1)
            bel = rng.uniform(-1, 1)
            
            # "Estimated" Opponent (True + Error)
            angle = rng.uniform(0, 2*np.pi)
            att_est = np.clip(att + err_mag * np.cos(angle), -1, 1)
            bel_est = np.clip(bel + err_mag * np.sin(angle), -1, 1)
            
            # Generate Game
            A, B = random_bimatrix_game(n_actions, rng)
            
            # -------------------------------------------------
            # 1. Opponent moves based on TRUE params
            # -------------------------------------------------
            A_real, B_real = make_modified_game(A, B, att_row=bel, att_col=att)
            game_real = nash.Game(A_real, B_real)
            
            # Opponent uses a random Nash starting label
            nash_method = rng.integers(0, 2 * n_actions) 
            
            sol = solve_robust(game_real, nash_method)
            
            if sol is None: continue
            _, strategy = sol # We want column strategy (opponent)
            
            # --- GUARD CLAUSES ---
            strategy = strategy.flatten()
            if strategy.shape[0] != n_actions: continue # Wrong size (Fixes IndexError)
            if strategy.sum() <= 1e-9: continue
            strategy /= strategy.sum()
            
            # Pick the move
            move = rng.choice(len(strategy), p=strategy)
            
            # -------------------------------------------------
            # 2. We predict based on ESTIMATED params
            # -------------------------------------------------
            A_est, B_est = make_modified_game(A, B, att_row=bel_est, att_col=att_est)
            game_est = nash.Game(A_est, B_est)
            
            sol_est = solve_robust(game_est, nash_method)
            
            if sol_est is None: continue
            _, strat_est = sol_est
            
            # --- GUARD CLAUSES ---
            strat_est = strat_est.flatten()
            if strat_est.shape[0] != n_actions: continue # Wrong size
            if strat_est.sum() <= 1e-9: continue
            strat_est /= strat_est.sum()
            
            # Check if move index is valid for this strategy array
            if move >= len(strat_est): continue
            
            prob_assigned = strat_est[move]
            
            # -------------------------------------------------
            # 3. Binning
            # -------------------------------------------------
            p_idx = min(int(prob_assigned * 5), 4) 
            
            # Calculate cooperation level
            denom = (np.sqrt(att_est**2 + 1) * np.sqrt(bel_est**2 + 1))
            coop = (att_est + bel_est) / denom
            
            c_idx = min(int((coop + 1) * 2.5), 4)
            
            T[p_idx, c_idx, l_idx] += 1
            valid_samples += 1

    # 3. Normalize
    sums = T.sum(axis=0, keepdims=True)
    T = T / np.where(sums > 0, sums, 1)
    
    np.savez("lookup_table.npz", T=T, prob_edges=prob_bins, coop_edges=coop_bins, error_levels=error_levels)
    print("Done! Saved 'lookup_table.npz'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    
    generate_simple_table(n_samples=args.samples)