from __future__ import annotations
import numpy as np

def compute_metrics(
    est_attitudes,
    est_beliefs,
    *,
    att_true: float | None = None,
    bel_true: float | None = None,
    est_errors=None,
    prob_est=None,
    prob_true=None,
    eps: float = 1e-12,
):
    """
    Compute and print key metrics (single run).
    - True error: Euclidean distance between (att_true, bel_true) and (att_est, bel_est).
    - Predictive accuracy: ratio prob_est / prob_true (clipped to [0,1] for plotting stability).
    """
    est_att = np.asarray(est_attitudes, dtype=float)
    est_bel = np.asarray(est_beliefs, dtype=float)
    T = len(est_att)

    print("\n[METRICS] Run summary")
    print(f"[METRICS] T={T}")

    out = {"T": T}

    # True error curve (Figure 3 "True Error")
    if att_true is not None and bel_true is not None:
        true_err = np.sqrt((att_true - est_att) ** 2 + (bel_true - est_bel) ** 2)
        out["true_error"] = true_err
        print(f"[METRICS] True params: att_true={att_true:.3f}, bel_true={bel_true:.3f}")
        print(f"[METRICS] Final estimates: att_est={est_att[-1]:.3f}, bel_est={est_bel[-1]:.3f}")
        print(f"[METRICS] Final true error: {true_err[-1]:.6f}")
        print(f"[METRICS] Mean true error:  {float(true_err.mean()):.6f}")

    # Estimated error curve (Figure 3 "Estimated Error") if you log it
    if est_errors is not None:
        est_err = np.asarray(est_errors, dtype=float)
        out["estimated_error"] = est_err
        print(f"[METRICS] Final estimated error: {est_err[-1]:.6f}")
        print(f"[METRICS] Mean estimated error:  {float(est_err.mean()):.6f}")

    # Predictive accuracy curve (Figure 3 dotted line) if you log prob_est & prob_true
    if prob_est is not None and prob_true is not None:
        p_est = np.asarray(prob_est, dtype=float)
        p_true = np.asarray(prob_true, dtype=float)
        ratio = p_est / (p_true + eps)
        ratio = np.clip(ratio, 0.0, 1.0)  # keeps it in [0,1] like the paper's plot
        out["predictive_accuracy"] = ratio
        print(f"[METRICS] Final predictive accuracy (ratio): {ratio[-1]:.6f}")
        print(f"[METRICS] Mean predictive accuracy (ratio):  {float(ratio.mean()):.6f}")

    return out
