from __future__ import annotations
import numpy as np
import numpy as np

def make_modified_game(A: np.ndarray, B: np.ndarray, att_row: float, att_col: float):
    """
    Génère les matrices de gains modifiées en fonction de l'attitude de chaque joueur.
    
    A : np.ndarray - Matrice des gains originaux pour le joueur en ligne (Row player)
    B : np.ndarray - Matrice des gains originaux pour le joueur en colonne (Column player)
    att_row : float - Attitude de l'agent ligne (alpha_i)
    att_col : float - Attitude de l'agent colonne (alpha_j)
    
    Retourne :
    A_mod : Matrice des gains perçus par le joueur ligne
    B_mod : Matrice des gains perçus par le joueur colonne
    """
    
    # Pour le joueur ligne (Row), son utilité est : 
    # son gain (A) + son attitude envers l'autre * le gain de l'autre (B)
    A_mod = A + (att_row * B)
    
    # Pour le joueur colonne (Col), son utilité est : 
    # son gain (B) + son attitude envers l'autre * le gain de l'autre (A)
    B_mod = B + (att_col * A)
    
    return A_mod, B_mod