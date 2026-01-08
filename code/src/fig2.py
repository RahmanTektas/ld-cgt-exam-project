from __future__ import annotations

import argparse
import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import nashpy as nash
from game import random_bimatrix_game
from modified_game import make_modified_game
from nash import solve_robust, safe_probvec

N_ACTIONS = 16
N_GAMES = 2000


# ---------- cooperation function (paper Eq. 3) ----------
def cooperation(att: float, bel: float) -> float:
    return (att + bel) / (np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0))


def smooth_moving_average(y, window=7):
    window = int(window)
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window, dtype=float) / window

    y_pad = np.pad(np.asarray(y, dtype=float), pad_width=pad, mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")  # length preserved
    return y_smooth


def _k_to_att_bel(k_target: float) -> tuple[float, float]:
    """Given k in [-1,1], pick a symmetric (att=bel=a) that matches it."""
    k = float(k_target)
    if abs(k) < 1e-12:
        a = 0.0
    else:
        disc = max(0.0, 1.0 - k * k)  # safety
        a = k / (1.0 + np.sqrt(disc))

    # keep it inside [-1,1]
    a = float(np.clip(a, -1.0, 1.0))
    return a, a


def _eval_point(idx: int, k_target: float, n_games: int, n_actions: int, base_seed: int, n_points: int) -> tuple[int, float, int]:
    """
    Compute y[idx] for one cooperation level:
      y = average over games of E_{m~sigma}[sigma[m]] = sum_i sigma_i^2
    Returns (idx, y_value, count_used).
    """
    # Make results independent of number of workers / chunking
    rng = np.random.default_rng(base_seed + 1000 * idx)

    att, bel = _k_to_att_bel(k_target)

    # ---- DEBUG: only for endpoints ----
    is_edge = idx in (0, 1, n_points - 2, n_points - 1)
    if is_edge:
        print("\n[DEBUG-POINT-BEGIN]")
        print(f"  idx={idx}/{n_points-1}  k={k_target:.6f}")
        print(f"  mapped att=bel=a -> a={att:.6f}  (clipped in [-1,1])")

        # quick check of modified game magnitude on one random test game
        A_test, B_test = random_bimatrix_game(n_actions, rng=rng)
        A2t, B2t = make_modified_game(A_test, B_test, att_row=bel, att_col=att)

        print(f"  A_test range: {A_test.min():.3g} .. {A_test.max():.3g}")
        print(f"  B_test range: {B_test.min():.3g} .. {B_test.max():.3g}")
        print(f"  A2 (modified) range: {A2t.min():.3g} .. {A2t.max():.3g}")
        print(f"  B2 (modified) range: {B2t.min():.3g} .. {B2t.max():.3g}")

    acc = 0.0
    cnt = 0

    for g in range(n_games):
        # progress print ONLY for edge points to avoid spam
        if is_edge and (g in (0, 1, 2) or g in (n_games // 2,) or g == n_games - 1):
            print(f"  [DEBUG] idx={idx} k={k_target:+.3f} game {g+1}/{n_games}")

        A, B = random_bimatrix_game(n_actions, rng=rng)
        A2, B2 = make_modified_game(A, B, att_row=bel, att_col=att)

        label = int(rng.integers(0, 2 * n_actions))
        res = solve_robust(nash.Game(A2, B2), label)
        if res is None:
            continue

        _, sigma = res
        sigma = safe_probvec(sigma, n_actions)

        acc += float(np.sum(sigma * sigma))  # exact E[sigma[m]]
        cnt += 1

    y_val = float(acc / cnt) if cnt > 0 else 0.0

    # ---- DEBUG: summary for endpoints ----
    if is_edge:
        print("[DEBUG-POINT-END]")
        print(f"  idx={idx} k={k_target:.6f}  count_used={cnt}/{n_games}  y_val={y_val:.6f}")

    return idx, y_val, cnt


def _worker_chunk(args):
    """
    Worker for a chunk of indices.
    Returns (idxs, y_chunk, counts_chunk).
    """
    base_seed, idxs, coop_grid, n_games, n_actions, n_points = args

    idxs = np.asarray(idxs, dtype=int)
    y_chunk = np.zeros(len(idxs), dtype=float)
    c_chunk = np.zeros(len(idxs), dtype=int)

    for t, idx in enumerate(idxs):
        _, y_val, c_val = _eval_point(
            int(idx),
            float(coop_grid[int(idx)]),
            n_games=n_games,
            n_actions=n_actions,
            base_seed=base_seed,
            n_points=n_points,
        )
        y_chunk[t] = y_val
        c_chunk[t] = c_val

    return idxs, y_chunk, c_chunk


# ---------- compute expected observed probability ----------
def compute_fig2_curve(
    coop_grid: np.ndarray,
    n_games: int = 1000,
    n_actions: int = 16,
    seed: int = 0,
    n_workers: int = 0,
):
    coop_grid = np.asarray(coop_grid, dtype=float)
    y = np.zeros_like(coop_grid, dtype=float)
    counts = np.zeros_like(coop_grid, dtype=int)

    n_points = len(coop_grid)

    # Single-process (keeps tqdm)
    if n_workers <= 0:
        for idx, k_target in enumerate(tqdm(coop_grid, desc="Fig2 (single)")):
            _, y_val, c_val = _eval_point(
                idx,
                float(k_target),
                n_games,
                n_actions,
                seed,
                n_points,
            )
            y[idx] = y_val
            counts[idx] = c_val
        return y, counts

    # Multi-process: split indices across workers
    idx_all = np.arange(n_points, dtype=int)
    chunks = np.array_split(idx_all, n_workers)
    tasks = [(seed, chunk, coop_grid, n_games, n_actions, n_points) for chunk in chunks]

    with mp.Pool(processes=n_workers) as pool:
        for idxs, y_chunk, c_chunk in pool.map(_worker_chunk, tasks):
            y[idxs] = y_chunk
            counts[idxs] = c_chunk

    return y, counts


# ---------- plotting ----------
def plot_fig2(x, y, save_pdf: str | None = None):
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(x, y, "b-", linewidth=0.8)

    ax.set_title("Expected Probability of Observed Move Given Cooperation Level")
    ax.set_xlabel("Cooperation Level")
    ax.set_ylabel("Expected Observed Probability")

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.linspace(-0.999, 0.999, 11))
    ax.set_yticks(np.linspace(0, 1, 11))

    ax.grid(False)
    plt.tight_layout()

    if save_pdf is not None:
        outdir = os.path.dirname(save_pdf)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(save_pdf, format="pdf", bbox_inches="tight")
        print(f"[INFO] saved {save_pdf}")

    plt.show()


# ---------- main ----------
if __name__ == "__main__":
    mp.freeze_support()  # for Windows support

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=N_GAMES)
    ap.add_argument("--actions", type=int, default=N_ACTIONS)
    ap.add_argument("--points", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = no multiprocessing")
    ap.add_argument("--pdf", type=str, default=None, help="Optional path to save the plot as PDF")
    args = ap.parse_args()

    coop_grid = np.linspace(-0.999, 0.999, args.points)

    y, counts = compute_fig2_curve(
        coop_grid,
        n_games=args.games,
        n_actions=args.actions,
        seed=args.seed,
        n_workers=args.workers,
    )

    print("\n[INFO] mean samples per point:", float(counts.mean()))
    print("[INFO] min samples per point:", int(counts.min()))
    i_min = int(np.argmin(counts))
    print(f"[INFO] argmin count: idx={i_min}, k={coop_grid[i_min]:.6f}, count={int(counts[i_min])}")

    # You can also inspect the endpoints directly:
    print(f"[INFO] endpoint counts: k={coop_grid[0]:.6f} -> {counts[0]}/{args.games}, "
          f"k={coop_grid[-1]:.6f} -> {counts[-1]}/{args.games}")

    y_smooth = smooth_moving_average(y, window=7)
    plot_fig2(coop_grid, y_smooth, save_pdf=args.pdf)
    plot_fig2(coop_grid, y, save_pdf=args.pdf )       
    
