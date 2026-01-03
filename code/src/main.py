import numpy as np
from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm
from metrics import compute_metrics
import argparse
from tqdm import tqdm
from collections import Counter

from modified_game import make_modified_game
from nash import one_nash_equilibrium

import os
import json


def debug_particles(agent, label: str, t: int, every: int = 100) -> None:
    """Print particle statistics for debugging."""
    if not hasattr(agent, "particles"):
        return

    if t not in (0, 1, 2, 5, 10) and (t % every != 0):
        return

    p_att = np.array([p.att for p in agent.particles], dtype=float)
    p_bel = np.array([p.bel for p in agent.particles], dtype=float)

    print(f"\n[DEBUG t={t}] {label}")
    print(f"  Particle att: min={p_att.min():.3f}, max={p_att.max():.3f}, mean={p_att.mean():.3f}, std={p_att.std():.3f}")
    print(f"  Particle bel: min={p_bel.min():.3f}, max={p_bel.max():.3f}, mean={p_bel.mean():.3f}, std={p_bel.std():.3f}")


def run_stationary(T=1000, n_actions=16, seed=42, outdir="results"):
    """Experiment: one learning agent vs a stationary opponent."""
    print("[INFO] Starting stationary-opponent experiment")
    print(f"[INFO] seed={seed}, T={T}, n_actions={n_actions}")

    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 1)
    rng_opp = np.random.default_rng(seed + 2)

    # Opponent: fixed (stationary) parameters
    opp_att = rng_opp.normal(0.0, 1.0)
    opp_bel = rng_opp.normal(0.0, 1.0)
    opp_att = np.clip(opp_att, -1.0, 1.0)
    opp_bel = np.clip(opp_bel, -1.0, 1.0)
    opp_nash = int(rng_opp.integers(0, 2 * n_actions))

    print(f"[INFO] Stationary opponent: att_true={opp_att:.3f}, bel_true={opp_bel:.3f}, nash_true={opp_nash}")

    # Agent
    agent = CooperativeAgentAlgorithm(nb_actions=n_actions, rng=rng_agent)

    # Logs (time series)
    logs = {
        "t": [],
        "agent_move": [],
        "opp_move": [],
        "r_agent": [],
        "r_opp": [],
        "att_est_before": [],
        "bel_est_before": [],
        "att_est_after": [],
        "bel_est_after": [],
        "true_error": [],
        "prob_true": [],
        "prob_est": [],
        "error_est": [],
    }

    for t in tqdm(range(T)):
        # 1) New random game
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # Basic sanity checks
        assert A.shape == (n_actions, n_actions), f"A has shape {A.shape}, expected {(n_actions, n_actions)}"
        assert B.shape == (n_actions, n_actions), f"B has shape {B.shape}, expected {(n_actions, n_actions)}"
        assert np.isfinite(A).all() and np.isfinite(B).all(), "Game matrices contain NaN/Inf"

        # 2) Stationary opponent picks a move (model-based, not uniform random)
        A_opp, B_opp = make_modified_game(A, B, att_row=opp_bel, att_col=opp_att)
        _, sigma_col = one_nash_equilibrium(A_opp, B_opp, opp_nash)

        # Safety: ensure sigma_col is a valid probability vector
        sigma_col = np.asarray(sigma_col, dtype=float)
        if (not np.isfinite(sigma_col).all()) or (sigma_col.sum() <= 0.0):
            print(f"[WARN t={t}] Invalid opponent strategy; falling back to uniform.")
            sigma_col = np.ones(n_actions, dtype=float) / n_actions
        else:
            sigma_col = sigma_col / sigma_col.sum()

        opp_move = int(rng_opp.choice(n_actions, p=sigma_col))
        prob_true_t = float(sigma_col[opp_move])

        # 3) Agent picks a move
        agent.A, agent.B = A, B
        agent_move = int(agent.pick_move())

        # Make sure actions are in bounds
        assert 0 <= agent_move < n_actions, f"Agent move out of bounds: {agent_move}"
        assert 0 <= opp_move < n_actions, f"Opponent move out of bounds: {opp_move}"

        # 4) Payoffs
        r_agent = float(A[agent_move, opp_move])
        r_opp = float(B[agent_move, opp_move])
        assert np.isfinite(r_agent) and np.isfinite(r_opp), "Payoff contains NaN/Inf"

        # --- Estimates BEFORE update ---
        if hasattr(agent, "estimate_attitude") and hasattr(agent, "estimate_belief"):
            att_est_before = float(agent.estimate_attitude())
            bel_est_before = float(agent.estimate_belief())
        else:
            att_est_before = float("nan")
            bel_est_before = float("nan")

        # Log "estimated" probability of the observed opponent move (agent's current model)
        nash_est_before = Counter([p.nash for p in agent.particles]).most_common(1)[0][0]

        A_est, B_est = make_modified_game(A, B, att_row=bel_est_before, att_col=att_est_before)
        _, sigma_col_est = one_nash_equilibrium(A_est, B_est, nash_est_before)

        sigma_col_est = np.asarray(sigma_col_est, dtype=float)
        if (not np.isfinite(sigma_col_est).all()) or (sigma_col_est.sum() <= 0.0):
            sigma_col_est = np.ones(n_actions, dtype=float) / n_actions
        else:
            sigma_col_est = sigma_col_est / sigma_col_est.sum()

        prob_est_t = float(sigma_col_est[opp_move])

        if t in (0, 1, 2, 5, 10) or t % 100 == 0:
            print(f"\n[STEP t={t}]")
            print(f"  Actions: agent_move={agent_move}, opp_move={opp_move}")
            print(f"  Payoffs: r_agent={r_agent:.4f}, r_opp={r_opp:.4f}")
            print(f"  Estimates BEFORE update: att_est={att_est_before:.3f}, bel_est={bel_est_before:.3f}")
            print(f"  True opponent params:     att_true={opp_att:.3f}, bel_true={opp_bel:.3f}")
            print(f"  Probabilities: prob_true={prob_true_t:.6f}, prob_est={prob_est_t:.6f}")

        debug_particles(agent, label="BEFORE update_model()", t=t, every=100)

        # 5) Model update (particle filter): observe opponent move then update
        agent.m = opp_move
        error_est = agent.update_model()

        # --- Estimates AFTER update ---
        if hasattr(agent, "estimate_attitude") and hasattr(agent, "estimate_belief"):
            att_est_after = float(agent.estimate_attitude())
            bel_est_after = float(agent.estimate_belief())
        else:
            att_est_after = float("nan")
            bel_est_after = float("nan")

        if t in (0, 1, 2, 5, 10) or t % 100 == 0:
            print(f"  error_est returned by update_model(): {error_est}")
            print(f"  Estimates AFTER update:  att_est={att_est_after:.3f}, bel_est={bel_est_after:.3f}")

        debug_particles(agent, label="AFTER update_model()", t=t, every=100)

        # True error (Euclidean distance) between true and estimated opponent parameters
        te = float(np.sqrt((opp_att - att_est_after) ** 2 + (opp_bel - bel_est_after) ** 2))

        # Append logs for this step
        logs["t"].append(t)
        logs["agent_move"].append(agent_move)
        logs["opp_move"].append(opp_move)
        logs["r_agent"].append(r_agent)
        logs["r_opp"].append(r_opp)
        logs["att_est_before"].append(att_est_before)
        logs["bel_est_before"].append(bel_est_before)
        logs["att_est_after"].append(att_est_after)
        logs["bel_est_after"].append(bel_est_after)
        logs["true_error"].append(te)
        logs["prob_true"].append(prob_true_t)
        logs["prob_est"].append(prob_est_t)
        logs["error_est"].append(float(error_est) if error_est is not None else np.nan)

    # End-of-run summary prints
    if len(logs["t"]) > 0:
        print("\n[SUMMARY] Final estimates")
        print(f"  True opponent params: att_true={opp_att:.3f}, bel_true={opp_bel:.3f}")
        print(f"  Final estimates:      att_est={logs['att_est_after'][-1]:.3f}, bel_est={logs['bel_est_after'][-1]:.3f}")
        print(f"  Final true error:     {logs['true_error'][-1]:.6f}")
        print(f"  Avg true error:       {float(np.mean(logs['true_error'])):.6f}")

    # Save to NPZ (compressed)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"stationary_seed{seed}_A{n_actions}_T{T}.npz")

    meta = {
        "scenario": "stationary",
        "seed": int(seed),
        "T": int(T),
        "n_actions": int(n_actions),
        "opp_att_true": float(opp_att),
        "opp_bel_true": float(opp_bel),
        "opp_nash_true": int(opp_nash),
    }

    np.savez_compressed(
        out_path,
        **{k: np.asarray(v) for k, v in logs.items()},
        meta=json.dumps(meta),
    )

    print(f"[INFO] Saved run to: {out_path}")

    # Compute metrics (kept as-is)
    compute_metrics(
        logs["att_est_after"],
        logs["bel_est_after"],
        att_true=opp_att,
        bel_true=opp_bel,
        prob_true=logs["prob_true"],
        prob_est=logs["prob_est"],
        est_errors=logs["error_est"],
    )

    print("[INFO] Simulation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["stationary"], default="stationary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--n_actions", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    if args.scenario == "stationary":
        run_stationary(T=args.T, n_actions=args.n_actions, seed=args.seed, outdir=args.outdir)
