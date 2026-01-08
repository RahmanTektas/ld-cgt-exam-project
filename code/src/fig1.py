from __future__ import annotations

"""plotting script for Figure 1 .

This module computes an expected-payoff surface over two attitude
parameters by sampling many random bi-matrix games and solving a
Nash equilibrium for each modified game. The computed surface
can be saved to an NPZ file and optionally plotted.

We used multiprocessing to speed up the computation.
It can be used with '--workers n' where n is the number of parallel processes.

The main function is 'compute_fig1_surface' which returns the grid.
"""

import argparse
import logging
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import nashpy as nash
from game import random_bimatrix_game
from modified_game import make_modified_game
from nash import solve_robust, safe_probvec

N_GAMES = 1000
N_ACTIONS = 16
GRID_N = 20

logger = logging.getLogger(__name__)

def colored_wireframe(ax, X, Y, Z, cmap=cm.viridis, lw=0.8):
    """Draw a colored wireframe on a 3D Axes.

    The wireframe is drawn by creating short line segments along rows
    and columns and coloring each segment according to the mid-point
    Z value. 
    """
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

    # Ensure axes scale to the data extents
    ax.auto_scale_xyz([float(X.min()), float(X.max())],
                      [float(Y.min()), float(Y.max())],
                      [float(Z.min()), float(Z.max())])


# ---------- computation ----------
def _compute_chunk(args):
    """Compute aggregated payoffs for a chunk of random games.

    The function is intentionally small and self-contained so it can
    be used with multiprocessing.Pool. It returns the sum of payoffs
    and the count of successful solves for each grid cell.
    """
    (seed, n_games, n_actions, grid, label_mode) = args
    rng = np.random.default_rng(seed)

    grid_n = len(grid)
    payoff_sum = np.zeros((grid_n, grid_n), dtype=float)
    counts = np.zeros((grid_n, grid_n), dtype=np.int32)

    for game_idx in range(n_games):
        if(game_idx%50)==0:
            print(game_idx)
        logger.debug("Generating random game %d/%d (seed=%d)", game_idx + 1, n_games, seed)
        A, B = random_bimatrix_game(n_actions, rng=rng)

        for i, att_agent in enumerate(grid):
            for j, att_opp in enumerate(grid):
                A2, B2 = make_modified_game(A, B, att_row=float(att_agent), att_col=float(att_opp))

                if label_mode == "fixed0":
                    label = 0
                else:
                    # 'random' mode
                    label = int(rng.integers(0, 2 * n_actions))

                res = solve_robust(nash.Game(A2, B2), label)
                if res is None:
                    # solver failed or no robust solution found
                    continue

                sr, sc = res
                sr = safe_probvec(sr, n_actions)
                sc = safe_probvec(sc, n_actions)

                # payoff measured using the original A matrix (per paper)
                payoff = float(sr @ A @ sc)
                payoff_sum[i, j] += payoff
                counts[i, j] += 1

    return payoff_sum, counts


def compute_fig1_surface(
    n_games=1000,
    n_actions=16,
    grid_n=21,
    seed=0,
    n_workers=0,
    label_mode="random",
):
    """Compute the Figure 1 surface by sampling many random games.

    Returns a tuple (grid, avg_payoff, counts) where `grid` is the
    attitude parameter vector, `avg_payoff` is a (grid_n, grid_n)
    array of mean payoffs, and `counts` contains the number of
    successful solves that contributed to each cell.
    """
    grid = np.linspace(-1.0, 1.0, grid_n)

    if n_workers <= 0:
        # single-process
        payoff_sum, counts = _compute_chunk((seed, n_games, n_actions, grid, label_mode))
    else:
        # multi-process by splitting the total games across workers
        chunks = np.array_split(np.arange(n_games), n_workers)
        tasks = []
        for w, idxs in enumerate(chunks):
            tasks.append((seed + 1000 * w, len(idxs), n_actions, grid, label_mode))

        with mp.Pool(processes=n_workers) as pool:
            out = pool.map(_compute_chunk, tasks)

        payoff_sum = np.zeros((grid_n, grid_n), dtype=float)
        counts = np.zeros((grid_n, grid_n), dtype=np.int32)
        for ps, cs in out:
            payoff_sum += ps
            counts += cs

    avg_payoff = np.divide(payoff_sum, counts, out=np.zeros_like(payoff_sum), where=(counts > 0))
    return grid, avg_payoff, counts


def plot_fig1(grid, avg_payoff):
    """Render the average-payoff surface.

    The `indexing='ij'` meshgrid call keeps the first axis aligned
    with the agent attitude and the second with the opponent.
    """
    X, Y = np.meshgrid(grid, grid, indexing="ij")

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    colored_wireframe(ax, X, Y, avg_payoff, cmap=cm.viridis, lw=0.8)

    ax.set_title("Expected payoff with full knowledge of attitude", pad=14)
    ax.set_xlabel("Agent Attitude", labelpad=12)
    ax.set_ylabel("Opponent Attitude", labelpad=12)
    ax.set_zlabel("Agent Score", labelpad=10)

    # flip x-axis so -1 appears on the left as in the original figure
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
    parser = argparse.ArgumentParser(description="Compute and plot Fig.1 payoff surface")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="0 = no multiprocessing")
    parser.add_argument("--label_mode", choices=["random", "fixed0"], default="random")
    parser.add_argument("--save", type=str, default="fig1_surface.npz")
    args = parser.parse_args()

    # configure simple logging to stdout for convenience
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    grid, avg_payoff, counts = compute_fig1_surface(
        n_games=N_GAMES,
        n_actions=N_ACTIONS,
        grid_n=GRID_N,
        seed=args.seed,
        n_workers=args.workers,
        label_mode=args.label_mode,
    )

    np.savez(args.save, grid=grid, Z=avg_payoff, C=counts,
             games=N_GAMES, actions=N_ACTIONS, grid_n=GRID_N,
             seed=args.seed, label_mode=args.label_mode)

    logger.info("saved %s", args.save)
    logger.info("mean count per cell: %.1f (should be close to %d if few failures)", counts.mean(),N_GAMES)


    plot_fig1(grid, avg_payoff)
