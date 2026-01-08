# src/main.py
from __future__ import annotations

import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

import nashpy as nash

from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm
from metrics import compute_metrics
from modified_game import make_modified_game
from nash import solve_robust


def safe_sigma(sigma: np.ndarray, n_actions: int) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float).flatten()
    if (sigma.shape[0] != n_actions) or (not np.isfinite(sigma).all()) or sigma.sum() <= 0.0:
        return np.ones(n_actions, dtype=float) / n_actions
    sigma = np.clip(sigma, 0.0, None)
    s = float(sigma.sum())
    return sigma / s if s > 0.0 else (np.ones(n_actions, dtype=float) / n_actions)


def run_stationary(
    T: int = 100,
    n_actions: int = 16,
    seed: int = 42,
    outdir: str = "results",
    lookup_path: str = "lookup_table.npz",
    quiet: bool = False,
) -> str:
    if not quiet:
        print(f"[INFO] Stationary run | seed={seed} | T={T} | A={n_actions}")

    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 1)
    rng_opp = np.random.default_rng(seed + 2)

    # Opponent: fixed parameters
    opp_att = float(np.clip(rng_opp.normal(0.0, 1.0), -1.0, 1.0))
    opp_bel = float(np.clip(rng_opp.normal(0.0, 1.0), -1.0, 1.0))
    opp_nash = int(rng_opp.integers(0, 2 * n_actions))

    # Resolve lookup path relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(lookup_path):
        lookup_path = os.path.join(script_dir, lookup_path)

    agent = CooperativeAgentAlgorithm(
        nb_actions=n_actions,
        rng=rng_agent,
        lookup_path=lookup_path,
    )

    # Keep only what we need for analysis + fig3
    logs = {
        "t": [],
        "att_est_after": [],
        "bel_est_after": [],
        "true_error": [],
        "prob_true": [],
        "prob_est": [],
        "error_est": [],
    }

    it = tqdm(range(T), desc="Simulating", disable=quiet)

    for t in it:
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # --- Opponent move (true params) ---
        A_opp, B_opp = make_modified_game(A, B, att_row=opp_bel, att_col=opp_att)
        try:
            sol_true = solve_robust(nash.Game(A_opp, B_opp), opp_nash)
            if sol_true is None:
                raise ValueError("solve_robust returned None")
            _, sigma_col_true = sol_true
            sigma_col_true = safe_sigma(sigma_col_true, n_actions)
        except Exception:
            sigma_col_true = np.ones(n_actions, dtype=float) / n_actions

        opp_move = int(rng_opp.choice(n_actions, p=sigma_col_true))
        prob_true_t = float(sigma_col_true[opp_move])

        # --- Agent move ---
        agent.A, agent.B = A, B
        _ = int(agent.pick_move())  # not needed for fig3

        # --- prob_est as in paper's "error update" step ---
        att_est_before = float(agent.estimate_attitude())
        bel_est_before = float(agent.estimate_belief())
        nash_est_before = Counter([p.nash for p in agent.particles]).most_common(1)[0][0]

        att_agent_for_error = float(np.clip(bel_est_before, -1.0, 1.0))  # att_agent = bel_opp (estimated)
        att_opp_for_error = float(np.clip(att_est_before, -1.0, 1.0))    # estimated opponent attitude

        A_est, B_est = make_modified_game(A, B, att_row=att_agent_for_error, att_col=att_opp_for_error)

        try:
            sol_est = solve_robust(nash.Game(A_est, B_est), int(nash_est_before))
            if sol_est is None:
                raise ValueError("solve_robust returned None")
            _, sigma_col_est = sol_est
            sigma_col_est = safe_sigma(sigma_col_est, n_actions)
        except Exception:
            sigma_col_est = np.ones(n_actions, dtype=float) / n_actions

        prob_est_t = float(sigma_col_est[opp_move])

        # --- Update particle filter with observed move ---
        agent.m = opp_move
        error_est = agent.update_model()

        att_est_after = float(agent.estimate_attitude())
        bel_est_after = float(agent.estimate_belief())

        te = float(np.sqrt((opp_att - att_est_after) ** 2 + (opp_bel - bel_est_after) ** 2))

        logs["t"].append(t)
        logs["att_est_after"].append(att_est_after)
        logs["bel_est_after"].append(bel_est_after)
        logs["true_error"].append(te)
        logs["prob_true"].append(prob_true_t)
        logs["prob_est"].append(prob_est_t)
        logs["error_est"].append(float(error_est) if error_est is not None else np.nan)

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

    if not quiet:
        print(f"[INFO] Saved: {out_path}")
        print(f"[INFO] Final true error: {logs['true_error'][-1]:.4f}")

    # Optional: keep metrics printout
    compute_metrics(
        logs["att_est_after"],
        logs["bel_est_after"],
        att_true=opp_att,
        bel_true=opp_bel,
        prob_true=logs["prob_true"],
        prob_est=logs["prob_est"],
        est_errors=logs["error_est"],
    )

    return out_path
