from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import nashpy as nash

from src.game import random_bimatrix_game
from src.modified_game import make_modified_game
from src.nash import solve_robust, safe_probvec
from src.agent import CooperativeAgentAlgorithm  # <-- make sure your fixed agent is in src/agent.py


# ----------------------------
# Helpers
# ----------------------------
def cooperation(att: float, bel: float) -> float:
    denom = np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0)
    if denom <= 0:
        return 0.0
    return float((att + bel) / denom)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_sigma(sigma: np.ndarray, n_actions: int) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float).flatten()
    if sigma.shape[0] != n_actions or (not np.isfinite(sigma).all()):
        return np.ones(n_actions, dtype=float) / n_actions
    sigma = np.clip(sigma, 0.0, None)
    s = float(sigma.sum())
    return sigma / s if s > 0 else (np.ones(n_actions, dtype=float) / n_actions)


# ----------------------------
# One self-play run (one seed)
# ----------------------------
def run_selfplay_once(
    seed: int,
    T: int = 1000,
    n_actions: int = 16,
    lookup_path: str = "lookup_table.npz",
    outdir: str = "results",
) -> str:
    os.makedirs(outdir, exist_ok=True)

    rng_env = np.random.default_rng(seed)
    rng_A = np.random.default_rng(seed + 1)
    rng_B = np.random.default_rng(seed + 2)

    # Two learning agents (same algorithm)
    agentA = CooperativeAgentAlgorithm(rng=rng_A, nb_actions=n_actions, lookup_path=lookup_path)
    agentB = CooperativeAgentAlgorithm(rng=rng_B, nb_actions=n_actions, lookup_path=lookup_path)

    # Logs for figure 4
    avg_payoff = np.zeros(T, dtype=float)
    coop_level = np.zeros(T, dtype=float)

    for t in range(T):
        # New random game each round (paper: “randomly generated games” repeatedly)
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # Both observe same underlying game
        agentA.observe_game(A, B)  # row payoff A, col payoff B
        agentB.observe_game(B, A)  # IMPORTANT: swap perspective for agentB (so it’s symmetric)

        # Pick moves
        a_move = agentA.pick_move()
        b_move = agentB.pick_move()

        # Payoffs in underlying game:
        # agentA is "row", agentB is "col"
        rA = float(A[a_move, b_move])
        rB = float(B[a_move, b_move])

        avg_payoff[t] = 0.5 * (rA + rB)

        # Each agent observes opponent move
        agentA.observe_opponent_move(b_move)
        agentB.observe_opponent_move(a_move)

        # Update both
        agentA.update_model()
        agentB.update_model()

        # Cooperation level: compute paper's coop metric for each agent and average them
        coopA = cooperation(agentA.estimate_attitude(), agentA.estimate_belief())
        coopB = cooperation(agentB.estimate_attitude(), agentB.estimate_belief())
        coop_level[t] = 0.5 * (coopA + coopB)

    out_path = os.path.join(outdir, f"selfplay_seed{seed}_A{n_actions}_T{T}.npz")
    np.savez_compressed(
        out_path,
        avg_payoff=avg_payoff,
        coop_level=coop_level,
        seed=seed,
        T=T,
        n_actions=n_actions,
        lookup_path=str(lookup_path),
    )
    return out_path


# ----------------------------
# Aggregate and plot
# ----------------------------
def aggregate_selfplay(
    seeds: list[int],
    T: int,
    n_actions: int,
    outdir: str,
) -> tuple[np.ndarray, np.ndarray]:
    payoffs = []
    coops = []

    for s in seeds:
        path = os.path.join(outdir, f"selfplay_seed{s}_A{n_actions}_T{T}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing run file: {path}")

        data = np.load(path)
        payoffs.append(np.asarray(data["avg_payoff"], dtype=float))
        coops.append(np.asarray(data["coop_level"], dtype=float))

    payoffs = np.stack(payoffs, axis=0)  # (R,T)
    coops = np.stack(coops, axis=0)      # (R,T)

    mean_payoff = payoffs.mean(axis=0)
    mean_coop = coops.mean(axis=0)
    return mean_payoff, mean_coop


def plot_fig4(mean_payoff: np.ndarray, mean_coop: np.ndarray, save_pdf: str | None = None) -> None:
    T = len(mean_payoff)
    t = np.arange(1, T + 1)

    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)

    # Blue dotted payoff
    ax.plot(
        t, mean_payoff,
        color="b",
        linewidth=0.9,
        linestyle=":",
        dashes=(1, 4),
        label="Average payoff",
    )

    # Red solid cooperation
    ax.plot(
        t, mean_coop,
        color="r",
        linewidth=0.9,
        linestyle="-",
        label="Cooperation level",
    )

    ax.set_title("Achieving cooperation using reciprocation")
    ax.set_xlabel("Time")

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(False)

    leg = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.94, 0.12),
        frameon=True,
        framealpha=1.0,
        fancybox=False,
        borderpad=0.8,
        handlelength=3,
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    if save_pdf is not None:
        ensure_parent_dir(save_pdf)
        fig.savefig(save_pdf, format="pdf", bbox_inches="tight")
        print(f"[INFO] Saved PDF: {save_pdf}")

    plt.show()


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--n_actions", type=int, default=16)
    ap.add_argument("--runs", type=int, default=100, help="number of self-play runs (paper uses 100)")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--lookup", type=str, default="lookup_table.npz")
    ap.add_argument("--pdf", type=str, default=None, help="path to save PDF, e.g. report/fig4.pdf")
    ap.add_argument("--skip_run", action="store_true", help="only aggregate+plot existing files")
    args = ap.parse_args()

    # Resolve lookup path relative to project root if needed
    # (Assumes script is in src/plots; project root is ../../)
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    lookup_path = args.lookup
    if not os.path.isabs(lookup_path):
        lookup_path = os.path.join(project_root, lookup_path)

    seeds = list(range(args.seed0, args.seed0 + args.runs))

    if not args.skip_run:
        print(f"[INFO] Running self-play for {len(seeds)} runs...")
        for s in tqdm(seeds, desc="Self-play runs"):
            out_path = os.path.join(args.outdir, f"selfplay_seed{s}_A{args.n_actions}_T{args.T}.npz")
            if os.path.exists(out_path):
                continue
            run_selfplay_once(
                seed=s,
                T=args.T,
                n_actions=args.n_actions,
                lookup_path=lookup_path,
                outdir=args.outdir,
            )

    print("[INFO] Aggregating...")
    mean_payoff, mean_coop = aggregate_selfplay(
        seeds=seeds,
        T=args.T,
        n_actions=args.n_actions,
        outdir=args.outdir,
    )

    plot_fig4(mean_payoff, mean_coop, save_pdf=args.pdf)


if __name__ == "__main__":
    main()
