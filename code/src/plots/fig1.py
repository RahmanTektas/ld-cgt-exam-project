import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def colored_wireframe(ax, X, Y, Z, cmap=cm.jet, lw=0.8):
    """
    Wireframe whose *lines* are colored by Z (height), like the reference figure.
    X = Agent Attitude grid
    Y = Opponent Attitude grid
    Z = Agent Score grid
    """
    norm = colors.Normalize(vmin=float(np.min(Z)), vmax=float(np.max(Z)))

    # Row lines (fixed row i, varying column j)
    row_segments = []
    row_colors = []
    for i in range(Z.shape[0]):
        pts = np.stack([X[i, :], Y[i, :], Z[i, :]], axis=1)          # (n, 3)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)                 # (n-1, 2, 3)
        row_segments.append(segs)
        zmid = (Z[i, :-1] + Z[i, 1:]) / 2.0
        row_colors.append(cmap(norm(zmid)))

    row_segments = np.concatenate(row_segments, axis=0)
    row_colors = np.concatenate(row_colors, axis=0)
    ax.add_collection3d(Line3DCollection(row_segments, colors=row_colors, linewidths=lw))

    # Column lines (fixed column j, varying row i)
    col_segments = []
    col_colors = []
    for j in range(Z.shape[1]):
        pts = np.stack([X[:, j], Y[:, j], Z[:, j]], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        col_segments.append(segs)
        zmid = (Z[:-1, j] + Z[1:, j]) / 2.0
        col_colors.append(cmap(norm(zmid)))

    col_segments = np.concatenate(col_segments, axis=0)
    col_colors = np.concatenate(col_colors, axis=0)
    ax.add_collection3d(Line3DCollection(col_segments, colors=col_colors, linewidths=lw))

    # Autoscale for collections
    ax.auto_scale_xyz([float(X.min()), float(X.max())],
                      [float(Y.min()), float(Y.max())],
                      [float(Z.min()), float(Z.max())])


# ------------------------------------------------------------
# 1) GRID (PROPER SEMANTICS)
#    X-axis  = Agent Attitude
#    Y-axis  = Opponent Attitude
# ------------------------------------------------------------
n = 21
agent_att = np.linspace(-1, 1, n)
opp_att   = np.linspace(-1, 1, n)

# indexing="xy" ensures:
#   X varies along columns, Y varies along rows (standard Cartesian)
AGENT, OPP = np.meshgrid(agent_att, opp_att, indexing="xy")


# ------------------------------------------------------------
# 2) YOUR PAYOFF SURFACE
#    Replace this with your actual expected payoff computation.
#    IMPORTANT: payoff_surface(agent, opp) in THIS order.
# ------------------------------------------------------------
def payoff_surface(agent, opp):
    # Placeholder that produces a smooth saturating surface in [0.5, 1.0]
    z = 0.5 + 0.5 * (1 / (1 + np.exp(-3*(opp + 0.2)))) * (1 / (1 + np.exp(-2*(agent - 0.1))))
    z += 0.03 * np.exp(-6*((agent + 0.1)**2 + (opp - 0.2)**2))
    return np.clip(z, 0.5, 1.0)

Z = payoff_surface(AGENT, OPP)


# ------------------------------------------------------------
# 3) PLOT (MATCH STYLE)
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6), dpi=100)  # 800x600
ax = fig.add_subplot(111, projection="3d")

colored_wireframe(ax, AGENT, OPP, Z, cmap=cm.jet, lw=0.8)

# Text (verbatim)
ax.set_title("Expected payoff with full knowledge of attitude", pad=14)
ax.set_xlabel("Agent Attitude", labelpad=12)
ax.set_ylabel("Opponent Attitude", labelpad=12)
ax.set_zlabel("Agent Score", labelpad=10)

# Limits + ticks like the reference
ax.set_xlim(1, -1)
ax.set_ylim(-1, 1)
ax.set_zlim(0.5, 1.0)

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Camera angle (close to the screenshot)
ax.view_init(elev=23, azim=-45)

# Dotted grey grid + white panes (Matlab-ish look)
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


# src/scripts/fig1.py
import multiprocessing as mp
mp.set_start_method("fork", force=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import solve_robust
import nashpy as nash


def expected_payoff_surface(n_games=1000, n_actions=16, grid_n=21, seed=0, label=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1.0, 1.0, grid_n)
    Z = np.zeros((grid_n, grid_n), dtype=float)
    C = np.zeros((grid_n, grid_n), dtype=float)  # counts (skips)

    for _ in range(n_games):
        print(f"[INFO] Simulating game {_ + 1}/{n_games}")
        A, B = random_bimatrix_game(n_actions, rng=rng)
        for i, att_agent in enumerate(grid):
            for j, att_opp in enumerate(grid):
                A2, B2 = make_modified_game(A, B, att_row=att_agent, att_col=att_opp)

                res = solve_robust(nash.Game(A2, B2), label)
                if res is None:
                    continue
                sr, sc = res
                sr = np.asarray(sr, float); sc = np.asarray(sc, float)

                # payoff in ORIGINAL game A
                payoff = float(sr @ A @ sc)
                Z[i, j] += payoff
                C[i, j] += 1

    Z = np.divide(Z, C, out=np.zeros_like(Z), where=(C > 0))
    return grid, Z

if __name__ == "__main__":
    grid, Z = expected_payoff_surface()

    X, Y = np.meshgrid(grid, grid, indexing="ij")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    ax.set_title("Expected payoff with full knowledge of attitude")
    ax.set_xlabel("Agent Attitude")
    ax.set_ylabel("Opponent Attitude")
    ax.set_zlabel("Agent Score")
    plt.show()
