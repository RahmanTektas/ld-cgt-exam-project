# src/metrics.py
from __future__ import annotations
import numpy as np

def err_distance(att_true: float, bel_true: float, att_est: float, bel_est: float) -> float:
    """ 
    Euclidean distance between true and estimated (attitude, belief) pairs.
    """
    return float(np.sqrt((att_true - att_est) ** 2 + (bel_true - bel_est) ** 2))

def cooperation(att_est: float, bel_est: float) -> float:
    """
      coop = (att + bel) / (sqrt(att^2 + 1) * sqrt(bel^2 + 1))
    """
    denom = np.sqrt(att_est * att_est + 1.0) * np.sqrt(bel_est * bel_est + 1.0)
    return float((att_est + bel_est) / denom)

def compute_metrics():
  print("TODO compute metrics")