from __future__ import annotations

import os
import glob
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")


def load_run(path: str) -> dict:
    data = np.load(path, allow_pickle=True)

    att = np.asarray(data["att_est_after"], dtype=float)
    bel = np.asarray(data["bel_est_after"], dtype=float)

    # This is the paper's "estimated error" proxy (your particle filter's err estimate).
    est_err = np.asarray(data["error_est"], dtype=float)

    p_true = np.asarray(data["prob_true"], dtype=float)
    p_est = np.asarray(data["prob_est"], dtype=float)

    meta_raw = data.get("meta", None)
    if meta_raw is None:
        raise ValueError(f"{path} has no 'meta' field (cannot get true opponent params).")

    meta = json.loads(str(meta_raw))
    att_true = float(meta["opp_att_true"])
    bel_true = float(meta["opp_bel_true"])

    return {
        "att": att,
        "bel": bel,
        "est_err": est_err,
        "p_true": p_true,
        "p_est": p_est,
        "att_true": att_true,
        "bel_true": bel_true,
        "path": path,
    }


def compute_fig3_series(
    runs: list[dict],
    smooth_window: int = 1,
    eps_true: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns mean time-series:
      estimated_error(t), true_error(t), predictive_accuracy(t)

    predictive_accuracy(t) = mean over runs of (p_est / p_true)
      - ignores steps where p_true is extremely tiny (otherwise ratio is meaningless)
      - clips ratio to [0, 1] for MATLAB-style plotting like the paper
    """
    if not runs:
        raise ValueError("No runs provided.")

    T = min(len(r["att"]) for r in runs)

    # Normalize true error to [0,1]:
    # max distance in [-1,1]^2 is sqrt(8)
    norm = np.sqrt(8.0)

    est_err_all = []
    true_err_all = []
    ratio_all = []

    for r in runs:
        att = r["att"][:T]
        bel = r["bel"][:T]
        est_err = r["est_err"][:T]

        att_true = r["att_true"]
        bel_true = r["bel_true"]

        # --- True error (normalized distance in parameter space) ---
        te = np.sqrt((att - att_true) ** 2 + (bel - bel_true) ** 2) / norm
        te = np.clip(te, 0.0, 1.0)

        # --- Predictive accuracy ratio ---
        # IMPORTANT: ignore points where p_true is too small.
        p_true = r["p_true"][:T]
        p_est = r["p_est"][:T]

        ratio = np.full(T, np.nan, dtype=float)
        mask = np.isfinite(p_true) & np.isfinite(p_est) & (p_true > eps_true) & (p_est >= 0.0)
        ratio[mask] = p_est[mask] / p_true[mask]

        # The paper's curve lives in [0,1] and approaches 1.
        # Cap extreme values (caused by tiny p_true or numerical weirdness).
        ratio = np.clip(ratio, 0.0, 1.0)

        # --- Estimated error cleanup ---
        # Your err can be >1 depending on your error_levels; plot wants [0,1].
        # Robust scaling: clip to 1 by default.
        est_err = np.asarray(est_err, float)
        est_err = np.clip(est_err, 0.0, 1.0)

        if smooth_window > 1:
            est_err = moving_average(est_err, smooth_window)
            te = moving_average(te, smooth_window)
            # for ratio, smooth NaNs safely
            ratio_filled = np.nan_to_num(ratio, nan=np.nanmean(ratio))
            ratio = moving_average(ratio_filled, smooth_window)
            ratio = np.clip(ratio, 0.0, 1.0)

        est_err_all.append(est_err)
        true_err_all.append(te)
        ratio_all.append(ratio)

    est_err_mean = np.mean(np.stack(est_err_all, axis=0), axis=0)
    true_err_mean = np.mean(np.stack(true_err_all, axis=0), axis=0)

    ratio_stack = np.stack(ratio_all, axis=0)
    ratio_mean = np.nanmean(ratio_stack, axis=0)
    ratio_mean = np.nan_to_num(ratio_mean, nan=0.0)

    return (
        np.clip(est_err_mean, 0.0, 1.0),
        np.clip(true_err_mean, 0.0, 1.0),
        np.clip(ratio_mean, 0.0, 1.0),
    )


def plot_fig3(est_err: np.ndarray, true_err: np.ndarray, ratio: np.ndarray, save_pdf: str | None = None) -> None:
    T = len(est_err)
    t = np.arange(1, T + 1)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(t, est_err, color="b", linewidth=1.0, label="Estimated Error")
    ax.plot(t, true_err, color="g", linewidth=1.0, linestyle="--", dashes=(6, 6), label="True Error")
    ax.plot(t, ratio, color="r", linewidth=1.0, linestyle=":", dashes=(1, 4), label="Predictive Accuracy")

    ax.set_title("Learning performance against stationary opponent")
    ax.set_xlabel("Time")

    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, T + 1, max(1, T // 10)))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(False)

    leg = ax.legend(
        loc="center right",
        frameon=True,
        framealpha=1.0,
        fancybox=False,
        borderpad=0.8,
        handlelength=3,
    )
    leg.get_frame().set_edgecolor("black")

    plt.tight_layout()

    if save_pdf is not None:
        ensure_parent_dir(save_pdf)
        fig.savefig(save_pdf, format="pdf", bbox_inches="tight")
        print(f"[INFO] Saved PDF: {save_pdf}")

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, required=True, help='e.g. "results/stationary_seed*_A16_T1000.npz"')
    ap.add_argument("--pdf", type=str, default=None, help="e.g. report/fig3.pdf")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window (try 5 or 10)")
    ap.add_argument("--eps_true", type=float, default=1e-6, help="ignore ratio where p_true <= eps_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    runs = [load_run(p) for p in paths]

    est_err, true_err, ratio = compute_fig3_series(
        runs,
        smooth_window=args.smooth,
        eps_true=args.eps_true,
    )

    print(f"[INFO] Loaded runs: {len(paths)}")
    plot_fig3(est_err, true_err, ratio, save_pdf=args.pdf)


if __name__ == "__main__":
    main()
