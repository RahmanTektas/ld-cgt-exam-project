import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm
from metrics import compute_metrics

from modified_game import make_modified_game
from nash import one_nash_equilibrium


def debug_particles(agent, label: str, t: int, every: int = 100) -> None:
    if not hasattr(agent, "particles"):
        return
    if t not in (0, 1, 2, 5, 10) and (t % every != 0):
        return

    p_att = np.array([p.att for p in agent.particles], dtype=float)
    p_bel = np.array([p.bel for p in agent.particles], dtype=float)

    print(f"\n[DEBUG t={t}] {label}")
    print(f"  Particle att: min={p_att.min():.3f}, max={p_att.max():.3f}, mean={p_att.mean():.3f}, std={p_att.std():.3f}")
    print(f"  Particle bel: min={p_bel.min():.3f}, max={p_bel.max():.3f}, mean={p_bel.mean():.3f}, std={p_bel.std():.3f}")


def safe_sigma(sigma: np.ndarray, n_actions: int) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    if (not np.isfinite(sigma).all()) or sigma.sum() <= 0.0:
        return np.ones(n_actions, dtype=float) / n_actions
    sigma = np.clip(sigma, 0.0, None)
    s = sigma.sum()
    return sigma / s if s > 0.0 else (np.ones(n_actions, dtype=float) / n_actions)


def run_stationary(T=1000, n_actions=16, seed=42, outdir="results", lookup_path="lookup_table.npz"):
    print("[INFO] Starting stationary-opponent experiment")
    print(f"[INFO] seed={seed}, T={T}, n_actions={n_actions}")

    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 1)
    rng_opp = np.random.default_rng(seed + 2)

    # Opponent: fixed parameters
    opp_att = float(np.clip(rng_opp.normal(0.0, 1.0), -1.0, 1.0))
    opp_bel = float(np.clip(rng_opp.normal(0.0, 1.0), -1.0, 1.0))
    opp_nash = int(rng_opp.integers(0, 2 * n_actions))

    print(f"[INFO] Stationary opponent: att_true={opp_att:.3f}, bel_true={opp_bel:.3f}, nash_true={opp_nash}")

    # --- IMPORTANT: resolve lookup path robustly ---
    # If lookup_path is relative, interpret it relative to the project root (where this file sits)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # assumes script is in src/scripts
    if not os.path.isabs(lookup_path):
        lookup_path = os.path.join(project_root, lookup_path)

    # Agent (loads lookup table)
    agent = CooperativeAgentAlgorithm(
        nb_actions=n_actions,
        rng=rng_agent,
        lookup_path=lookup_path,
    )

    print("[INFO] Lookup table path:", lookup_path)
    print("[INFO] Lookup enabled:", getattr(agent, "use_lookup", False))

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

    for t in tqdm(range(T), desc="Simulating"):
        # 1) New random game
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # 2) Opponent move
        A_opp, B_opp = make_modified_game(A, B, att_row=opp_bel, att_col=opp_att)

        # Lemke–Howson can be slow/hang on some games for larger n.
        # If your nash solver has issues, we’ll just fallback to uniform instead of freezing.
        try:
            _, sigma_col = one_nash_equilibrium(A_opp, B_opp, opp_nash)
            sigma_col = safe_sigma(sigma_col, n_actions)
        except KeyboardInterrupt:
            raise
        except Exception:
            sigma_col = np.ones(n_actions, dtype=float) / n_actions

        opp_move = int(rng_opp.choice(n_actions, p=sigma_col))
        prob_true_t = float(sigma_col[opp_move])

        # 3) Agent move
        agent.A, agent.B = A, B
        agent_move = int(agent.pick_move())

        # 4) Payoffs
        r_agent = float(A[agent_move, opp_move])
        r_opp = float(B[agent_move, opp_move])

        # Estimates BEFORE update
        att_est_before = float(agent.estimate_attitude())
        bel_est_before = float(agent.estimate_belief())

        nash_est_before = Counter([p.nash for p in agent.particles]).most_common(1)[0][0]
        A_est, B_est = make_modified_game(A, B, att_row=bel_est_before, att_col=att_est_before)
        try:
            _, sigma_col_est = one_nash_equilibrium(A_est, B_est, nash_est_before)
            sigma_col_est = safe_sigma(sigma_col_est, n_actions)
        except KeyboardInterrupt:
            raise
        except Exception:
            sigma_col_est = np.ones(n_actions, dtype=float) / n_actions

        prob_est_t = float(sigma_col_est[opp_move])

        if t in (0, 1, 2, 5, 10) or t % 100 == 0:
            print(f"\n[STEP t={t}]")
            print(f"  Actions: agent_move={agent_move}, opp_move={opp_move}")
            print(f"  Payoffs: r_agent={r_agent:.4f}, r_opp={r_opp:.4f}")
            print(f"  Estimates BEFORE update: att_est={att_est_before:.3f}, bel_est={bel_est_before:.3f}")
            print(f"  True opponent params:     att_true={opp_att:.3f}, bel_true={opp_bel:.3f}")
            print(f"  Probabilities: prob_true={prob_true_t:.6f}, prob_est={prob_est_t:.6f}")

        debug_particles(agent, label="BEFORE update_model()", t=t, every=100)

        # 5) Update
        agent.m = opp_move
        error_est = agent.update_model()

        att_est_after = float(agent.estimate_attitude())
        bel_est_after = float(agent.estimate_belief())

        if t in (0, 1, 2, 5, 10) or t % 100 == 0:
            print(f"  error_est returned by update_model(): {error_est}")
            print(f"  Estimates AFTER update:  att_est={att_est_after:.3f}, bel_est={bel_est_after:.3f}")

        debug_particles(agent, label="AFTER update_model()", t=t, every=100)

        te = float(np.sqrt((opp_att - att_est_after) ** 2 + (opp_bel - bel_est_after) ** 2))

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

    print("\n[SUMMARY] Final estimates")
    print(f"  True opponent params: att_true={opp_att:.3f}, bel_true={opp_bel:.3f}")
    print(f"  Final estimates:      att_est={logs['att_est_after'][-1]:.3f}, bel_est={logs['bel_est_after'][-1]:.3f}")
    print(f"  Final true error:     {logs['true_error'][-1]:.6f}")
    print(f"  Avg true error:       {float(np.mean(logs['true_error'])):.6f}")

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
        "lookup_path": str(lookup_path),
    }

    np.savez_compressed(out_path, **{k: np.asarray(v) for k, v in logs.items()}, meta=json.dumps(meta))
    print(f"[INFO] Saved run to: {out_path}")

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
    parser.add_argument("--n_actions", type=int, default=16)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--lookup", type=str, default="lookup_table.npz")
    args = parser.parse_args()

    if args.scenario == "stationary":
        run_stationary(
            T=args.T,
            n_actions=args.n_actions,
            seed=args.seed,
            outdir=args.outdir,
            lookup_path=args.lookup,
        )
