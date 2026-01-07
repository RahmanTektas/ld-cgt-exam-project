from __future__ import annotations

import os
import argparse
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import nashpy as nash
from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import solve_robust, safe_probvec


# ---------- plotting (your Matlab-ish colored wireframe) ----------
def colored_wireframe(ax, X, Y, Z, cmap=cm.jet, lw=0.8):
    norm = colors.Normalize(vmin=float(np.min(Z)), vmax=float(np.max(Z)))

    row_segments, row_colors = [], []
    for i in range(Z.shape[0]):
        pts = np.stack([X[i, :], Y[i, :], Z[i, :]], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        row_segments.append(segs)
        zmid = (Z[i, :-1] + Z[i, 1:]) / 2.0
        row_colors.append(cmap(norm(zmid)))

    ax.add_collection3d(Line3DCollection(
        np.concatenate(row_segments, axis=0),
        colors=np.concatenate(row_colors, axis=0),
        linewidths=lw,
    ))

    col_segments, col_colors = [], []
    for j in range(Z.shape[1]):
        pts = np.stack([X[:, j], Y[:, j], Z[:, j]], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        col_segments.append(segs)
        zmid = (Z[:-1, j] + Z[1:, j]) / 2.0
        col_colors.append(cmap(norm(zmid)))

    ax.add_collection3d(Line3DCollection(
        np.concatenate(col_segments, axis=0),
        colors=np.concatenate(col_colors, axis=0),
        linewidths=lw,
    ))

    ax.auto_scale_xyz([float(X.min()), float(X.max())],
                      [float(Y.min()), float(Y.max())],
                      [float(Z.min()), float(Z.max())])


# ---------- computation ----------
def _worker_chunk(args):
    """
    Worker computes partial sums over a chunk of games.
    Returns (Z_sum, C_count) arrays of shape (grid_n, grid_n).
    """
    (seed, n_games, n_actions, grid, label_mode) = args
    rng = np.random.default_rng(seed)

    grid_n = len(grid)
    Z_sum = np.zeros((grid_n, grid_n), dtype=float)
    C = np.zeros((grid_n, grid_n), dtype=np.int32)

    # precompute all (att_agent, att_opp) pairs indices for speed
    for _ in range(n_games):
        print("[DEBUG] Generating new random game " + str(_ + 1) + "/" + str(n_games))
        A, B = random_bimatrix_game(n_actions, rng=rng)

        for i, att_agent in enumerate(grid):
            for j, att_opp in enumerate(grid):
                A2, B2 = make_modified_game(A, B, att_row=float(att_agent), att_col=float(att_opp))

                if label_mode == "fixed0":
                    label = 0
                elif label_mode == "fixed":
                    label = int(rng.integers(0, 2 * n_actions))  # fixed-per-solve? (still random)
                else:
                    # "random" (recommended): random dropped label per solve
                    label = int(rng.integers(0, 2 * n_actions))

                res = solve_robust(nash.Game(A2, B2), label)
                if res is None:
                    continue

                sr, sc = res
                sr = safe_probvec(sr, n_actions)
                sc = safe_probvec(sc, n_actions)

                # payoff measured in ORIGINAL A (paper)
                payoff = float(sr @ A @ sc)
                Z_sum[i, j] += payoff
                C[i, j] += 1

    return Z_sum, C


def compute_fig1_surface(
    n_games=1000,
    n_actions=16,
    grid_n=21,
    seed=0,
    n_workers=0,
    label_mode="random",
):
    """
    Returns: grid (grid_n,), Z (grid_n,grid_n), C counts
    """
    grid = np.linspace(-1.0, 1.0, grid_n)

    if n_workers <= 0:
        # single-process
        Z_sum, C = _worker_chunk((seed, n_games, n_actions, grid, label_mode))
    else:
        # multi-process by splitting games
        chunks = np.array_split(np.arange(n_games), n_workers)
        tasks = []
        for w, idxs in enumerate(chunks):
            tasks.append((seed + 1000 * w, len(idxs), n_actions, grid, label_mode))

        with mp.Pool(processes=n_workers) as pool:
            out = pool.map(_worker_chunk, tasks)

        Z_sum = np.zeros((grid_n, grid_n), dtype=float)
        C = np.zeros((grid_n, grid_n), dtype=np.int32)
        for Zp, Cp in out:
            Z_sum += Zp
            C += Cp

    Z = np.divide(Z_sum, C, out=np.zeros_like(Z_sum), where=(C > 0))
    return grid, Z, C


def plot_fig1(grid, Z):
    # meshgrid with ij so X=agent attitude (rows), Y=opp attitude (cols) matches your computation
    X, Y = np.meshgrid(grid, grid, indexing="ij")

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    colored_wireframe(ax, X, Y, Z, cmap=cm.jet, lw=0.8)

    ax.set_title("Expected payoff with full knowledge of attitude", pad=14)
    ax.set_xlabel("Agent Attitude", labelpad=12)
    ax.set_ylabel("Opponent Attitude", labelpad=12)
    ax.set_zlabel("Agent Score", labelpad=10)

    ax.set_xlim(1, -1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0.5, 1.0)

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.view_init(elev=23, azim=-45)

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.8
        axis._axinfo["grid"]["color"] = (0.65, 0.65, 0.65, 1.0)

    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--actions", type=int, default=16)
    ap.add_argument("--grid", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = no multiprocessing")
    ap.add_argument("--label_mode", choices=["random", "fixed0"], default="random",
                    help="random = random dropped label per solve (recommended). fixed0 = always 0 (can bias).")
    ap.add_argument("--save", type=str, default="fig1_surface.npz")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    grid, Z, C = compute_fig1_surface(
        n_games=args.games,
        n_actions=args.actions,
        grid_n=args.grid,
        seed=args.seed,
        n_workers=args.workers,
        label_mode=args.label_mode,
    )

    np.savez(args.save, grid=grid, Z=Z, C=C,
             games=args.games, actions=args.actions, grid_n=args.grid,
             seed=args.seed, label_mode=args.label_mode)

    print(f"[INFO] saved {args.save}")
    print(f"[INFO] mean count per cell: {C.mean():.1f} (should be close to {args.games} if few failures)")

    if not args.no_plot:
        plot_fig1(grid, Z)
