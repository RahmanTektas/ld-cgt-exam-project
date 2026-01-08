from __future__ import annotations

import os
import glob
import json
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def moving_average_edge(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average with edge-padding (avoids artificial drops at the boundaries)."""
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    k = np.ones(w, dtype=float) / w
    x_pad = np.pad(x, pad_width=pad, mode="edge")
    return np.convolve(x_pad, k, mode="valid")  # same length as x


def load_run(path: str) -> dict:
    data = np.load(path, allow_pickle=True)

    att = np.asarray(data["att_est_after"], dtype=float)
    bel = np.asarray(data["bel_est_after"], dtype=float)

    # paper's estimated error proxy from particle filter
    est_err = np.asarray(data["error_est"], dtype=float)

    p_true = np.asarray(data["prob_true"], dtype=float)
    p_est = np.asarray(data["prob_est"], dtype=float)

    meta_raw = data.get("meta", None)
    if meta_raw is None:
        raise ValueError(f"{path} has no 'meta' field.")

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
      - ignores steps where p_true is extremely tiny
      - clips ratio to [0, 1] like the paper plot

    Notes:
      - true_error is normalized by sqrt(8) (max distance in [-1,1]^2)
      - est_err is also normalized by sqrt(8) so it is comparable on the same [0,1] scale
    """
    if not runs:
        raise ValueError("No runs provided.")

    T = min(len(r["att"]) for r in runs)
    norm = np.sqrt(8.0)  # max Euclidean distance in [-1,1]^2

    est_err_all = []
    true_err_all = []
    ratio_all = []

    for r in runs:
        att = np.asarray(r["att"][:T], dtype=float)
        bel = np.asarray(r["bel"][:T], dtype=float)

        # --- True error: normalized Euclidean distance in parameter space ---
        att_true = float(r["att_true"])
        bel_true = float(r["bel_true"])
        te = np.sqrt((att - att_true) ** 2 + (bel - bel_true) ** 2) / norm
        te = np.clip(te, 0.0, 1.0)

        # --- Estimated error: normalize to match the same [0,1] scale ---
        est_err = np.asarray(r["est_err"][:T], dtype=float)
        est_err = est_err / norm
        est_err = np.clip(est_err, 0.0, 1.0)

        # --- Predictive accuracy ratio (p_est / p_true), clipped to [0,1] ---
        p_true = np.asarray(r["p_true"][:T], dtype=float)
        p_est = np.asarray(r["p_est"][:T], dtype=float)

        ratio = np.full(T, np.nan, dtype=float)
        mask = (
            np.isfinite(p_true)
            & np.isfinite(p_est)
            & (p_true > eps_true)
            & (p_est >= 0.0)
        )
        ratio[mask] = p_est[mask] / p_true[mask]
        ratio = np.clip(ratio, 0.0, 1.0)

        # --- Optional smoothing (edge-padded moving average) ---
        if smooth_window > 1:
            est_err = moving_average_edge(est_err, smooth_window)
            te = moving_average_edge(te, smooth_window)

            ratio_filled = np.nan_to_num(ratio, nan=np.nanmean(ratio))
            ratio = moving_average_edge(ratio_filled, smooth_window)
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


# -----------------------------
# Parallel run generation
# -----------------------------
def _run_one_stationary(args: tuple[int, int, int, str, str, bool, bool]) -> str:
    """
    Worker: generate a single stationary run and return the output path.
    args = (seed, T, actions, outdir, lookup, quiet, force)
    """
    seed, T, actions, outdir, lookup, quiet, force = args

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"stationary_seed{seed}_A{actions}_T{T}.npz")

    if (not force) and os.path.exists(out_path):
        return out_path

    # Import inside the worker (safe with Windows spawn)
    from main import run_stationary

    run_stationary(
        T=T,
        n_actions=actions,
        seed=seed,
        outdir=outdir,
        lookup_path=lookup,
        quiet=quiet,
    )
    return out_path


def generate_runs(
    runs: int,
    seed0: int,
    T: int,
    actions: int,
    outdir: str,
    lookup: str,
    workers: int = 0,
    quiet: bool = True,
    force: bool = False,
) -> list[str]:
    """
    Generate multiple runs, optionally in parallel.
    """
    tasks = []
    for i in range(runs):
        seed = seed0 + i
        tasks.append((seed, T, actions, outdir, lookup, quiet, force))

    # Sequential
    if workers <= 0:
        paths = []
        for task in tqdm(tasks, desc="Generating runs (single)"):
            paths.append(_run_one_stationary(task))
        return paths

    # Parallel (Windows-friendly spawn)
    ctx = mp.get_context("spawn")
    paths: list[str] = []
    with ctx.Pool(processes=workers) as pool:
        for out_path in tqdm(pool.imap_unordered(_run_one_stationary, tasks), total=len(tasks), desc=f"Generating runs ({workers} workers)"):
            paths.append(out_path)

    return sorted(paths)


def main():
    mp.freeze_support()  # important on Windows

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true", help="Generate runs first (then plot).")
    ap.add_argument("--runs", type=int, default=10, help="Number of runs to generate (paper uses 100).")
    ap.add_argument("--seed0", type=int, default=0, help="Starting seed for generation.")
    ap.add_argument("--workers", type=int, default=0, help="0 = no multiprocessing; else number of processes")
    ap.add_argument("--force", action="store_true", help="Regenerate runs even if .npz already exists")

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--actions", type=int, default=16)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--lookup", type=str, default="lookup_table.npz")

    ap.add_argument("--pattern", type=str, default=None, help='e.g. "results/stationary_seed*_A16_T1000.npz"')
    ap.add_argument("--pdf", type=str, default=None, help="e.g. report/fig3.pdf")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window (try 5 or 10)")
    ap.add_argument("--eps_true", type=float, default=1e-6, help="ignore ratio where p_true <= eps_true")
    args = ap.parse_args()

    # 1) Generate runs if asked
    if args.gen:
        paths = generate_runs(
            runs=args.runs,
            seed0=args.seed0,
            T=args.T,
            actions=args.actions,
            outdir=args.outdir,
            lookup=args.lookup,
            workers=args.workers,
            quiet=True,
            force=args.force,
        )
    else:
        if args.pattern is None:
            raise SystemExit("Provide --pattern for plot-only mode, or use --gen to generate runs.")
        paths = sorted(glob.glob(args.pattern))
        if not paths:
            raise SystemExit(f"No files matched pattern: {args.pattern}")

    # 2) Load + aggregate
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
