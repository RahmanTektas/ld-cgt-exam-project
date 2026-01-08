from __future__ import annotations

import os
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm


# ----------------------------
# Helpers
# ----------------------------
def cooperation(att: float, bel: float) -> float:
    denom = np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0)
    if denom <= 0.0:
        return 0.0
    return float((att + bel) / denom)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


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
        # New random game each round
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # Both observe the same underlying game, but with swapped perspective
        agentA.observe_game(A, B)  # agentA is "row"
        agentB.observe_game(B, A)  # agentB is "row" in its own internal view

        # Pick moves
        a_move = agentA.pick_move()
        b_move = agentB.pick_move()

        # Payoffs in underlying game: agentA uses A, agentB uses B
        rA = float(A[a_move, b_move])
        rB = float(B[a_move, b_move])

        avg_payoff[t] = 0.5 * (rA + rB)

        # Observe opponent move then update
        agentA.observe_opponent_move(b_move)
        agentB.observe_opponent_move(a_move)

        agentA.update_model()
        agentB.update_model()

        # Cooperation metric (paper Eq.3) from each agent's current estimates
        coopA = cooperation(agentA.estimate_attitude(), agentA.estimate_belief())
        coopB = cooperation(agentB.estimate_attitude(), agentB.estimate_belief())
        coop_level[t] = 0.5 * (coopA + coopB)

    out_path = os.path.join(outdir, f"selfplay_seed{seed}_A{n_actions}_T{T}.npz")
    np.savez_compressed(
        out_path,
        avg_payoff=avg_payoff,
        coop_level=coop_level,
        seed=int(seed),
        T=int(T),
        n_actions=int(n_actions),
        lookup_path=str(lookup_path),
    )
    return out_path


# ----------------------------
# Parallel run generation
# ----------------------------
def _worker_run_selfplay(args: tuple[int, int, int, str, str, bool]) -> str:
    """
    Worker entry-point (Windows spawn-safe).
    args = (seed, T, n_actions, lookup_path, outdir, force)
    """
    seed, T, n_actions, lookup_path, outdir, force = args
    out_path = os.path.join(outdir, f"selfplay_seed{seed}_A{n_actions}_T{T}.npz")

    if (not force) and os.path.exists(out_path):
        return out_path

    return run_selfplay_once(
        seed=seed,
        T=T,
        n_actions=n_actions,
        lookup_path=lookup_path,
        outdir=outdir,
    )


def generate_selfplay_runs(
    seeds: list[int],
    T: int,
    n_actions: int,
    lookup_path: str,
    outdir: str,
    workers: int = 0,
    force: bool = False,
) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    tasks = [(s, T, n_actions, lookup_path, outdir, force) for s in seeds]

    # Sequential
    if workers <= 0:
        out_paths = []
        for task in tqdm(tasks, desc="Self-play runs (single)"):
            out_paths.append(_worker_run_selfplay(task))
        return sorted(out_paths)

    # Parallel (Windows-friendly)
    ctx = mp.get_context("spawn")
    out_paths: list[str] = []
    with ctx.Pool(processes=workers) as pool:
        for p in tqdm(
            pool.imap_unordered(_worker_run_selfplay, tasks),
            total=len(tasks),
            desc=f"Self-play runs ({workers} workers)",
        ):
            out_paths.append(p)

    return sorted(out_paths)


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

    # Payoff (blue dotted)
    ax.plot(
        t, mean_payoff,
        color="b",
        linewidth=0.9,
        linestyle=":",
        dashes=(1, 4),
        label="Average payoff",
    )

    # Cooperation (red solid)
    ax.plot(
        t, mean_coop,
        color="r",
        linewidth=0.9,
        linestyle="-",
        label="Cooperation level",
    )

    ax.set_title("Achieving cooperation using reciprocation")
    ax.set_xlabel("Time")

    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, T + 1, max(1, T // 10)))
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
    mp.freeze_support()  # Windows support

    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--n_actions", type=int, default=16)
    ap.add_argument("--runs", type=int, default=100, help="number of self-play runs (paper uses 100)")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")

    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--lookup", type=str, default="lookup_table.npz")
    ap.add_argument("--pdf", type=str, default=None, help="path to save PDF, e.g. report/fig4.pdf")

    ap.add_argument("--workers", type=int, default=0, help="0 = no multiprocessing; else number of processes")
    ap.add_argument("--force", action="store_true", help="regenerate even if .npz exists")
    ap.add_argument("--skip_run", action="store_true", help="only aggregate+plot existing files")
    args = ap.parse_args()

    # Resolve lookup path relative to project root if needed
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    lookup_path = args.lookup
    if not os.path.isabs(lookup_path):
        lookup_path = os.path.join(project_root, lookup_path)

    seeds = list(range(args.seed0, args.seed0 + args.runs))

    if not args.skip_run:
        print(f"[INFO] Generating {len(seeds)} self-play runs...")
        _ = generate_selfplay_runs(
            seeds=seeds,
            T=args.T,
            n_actions=args.n_actions,
            lookup_path=lookup_path,
            outdir=args.outdir,
            workers=args.workers,
            force=args.force,
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
