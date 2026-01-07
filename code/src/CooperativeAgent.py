# src/agent.py
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
import nashpy as nash

from src.modified_game import make_modified_game
from src.nash_solve import solve_robust, safe_probvec


@dataclass
class Particle:
    att: float  # opponent attitude estimate
    bel: float  # opponent belief about us estimate
    nash: int   # Lemke–Howson dropped label (opponent Nash-selection proxy)


class CooperativeAgentAlgorithm:
    def __init__(
        self,
        rng: np.random.Generator,
        nb_actions: int = 16,
        nb_particles: int = 100,
        reciprocation: float = 0.1,
        fab: float = 0.1,
        fnash: float = 0.05,
        error_levels: np.ndarray | None = None,
        p_error: np.ndarray | None = None,
        lookup_path: str | None = None,
    ):
        self.rng = rng
        self.nb_actions = int(nb_actions)
        self.n = int(nb_particles)
        self.r = float(reciprocation)
        self.fab = float(fab)
        self.fnash = float(fnash)

        self.use_lookup = False
        if lookup_path is not None:
            data = np.load(lookup_path, allow_pickle=True)
            self.T = np.asarray(data["T"], dtype=float)                # (J,K,L)
            self.prob_edges = np.asarray(data["prob_edges"], dtype=float)
            self.coop_edges = np.asarray(data["coop_edges"], dtype=float)

            if error_levels is None:
                self.error_levels = np.asarray(data["error_levels"], dtype=float)
            else:
                self.error_levels = np.asarray(error_levels, dtype=float)

            L = len(self.error_levels)
            if p_error is None:
                self.P_error = np.ones(L, dtype=float) / L
            else:
                self.P_error = np.asarray(p_error, dtype=float)
                s = float(self.P_error.sum())
                self.P_error = (self.P_error / s) if s > 0 else (np.ones(L, dtype=float) / L)

            self.use_lookup = True

        # particles
        self.particles: list[Particle] = []
        for _ in range(self.n):
            att = float(np.clip(self.rng.normal(0.0, 1.0), -1.0, 1.0))
            bel = float(np.clip(self.rng.normal(0.0, 1.0), -1.0, 1.0))
            nash_label = int(self.rng.integers(0, 2 * self.nb_actions))
            self.particles.append(Particle(att=att, bel=bel, nash=nash_label))

        # per-round stored values
        self.A = None
        self.B = None
        self.m = 0

        self.last_att_opp_est = 0.0
        self.last_bel_opp_est = 0.0
        self.last_nash_opp = 0

        self.last_att_agent_used = 0.0
        self.last_att_opp_used = 0.0

    # ---------- helpers ----------
    @staticmethod
    def cooperation(att: float, bel: float) -> float:
        denom = np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0)
        return float((att + bel) / denom)

    @staticmethod
    def bin_index(x: float, edges: np.ndarray) -> int:
        idx = np.searchsorted(edges, x, side="right") - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    @staticmethod
    def clip_att(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))

    def estimate_attitude(self) -> float:
        return float(np.mean([p.att for p in self.particles]))

    def estimate_belief(self) -> float:
        return float(np.mean([p.bel for p in self.particles]))

    def estimate_nash(self) -> int:
        return int(Counter([p.nash for p in self.particles]).most_common(1)[0][0])

    # ---------- API ----------
    def observe_game(self, A: np.ndarray, B: np.ndarray) -> None:
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)

    def pick_move(self) -> int:
        assert self.A is not None and self.B is not None

        # (a) estimate opponent params
        att_opp_est = self.clip_att(self.estimate_attitude())
        bel_opp_est = self.clip_att(self.estimate_belief())
        nash_opp = self.estimate_nash()

        self.last_att_opp_est = att_opp_est
        self.last_bel_opp_est = bel_opp_est
        self.last_nash_opp = int(nash_opp)

        # (b) reciprocation rule (paper)
        att_agent_used = self.clip_att(att_opp_est + self.r)
        # opponent's *used* attitude depends on opponent's belief about us
        att_opp_used = self.clip_att(bel_opp_est + self.r)

        self.last_att_agent_used = att_agent_used
        self.last_att_opp_used = att_opp_used

        # (c) build modified game with *used* attitudes
        A_mod, B_mod = make_modified_game(self.A, self.B, att_agent_used, att_opp_used)

        sol = solve_robust(nash.Game(A_mod, B_mod), nash_opp)
        if sol is None:
            sigma_row = np.ones(self.nb_actions) / self.nb_actions
            sigma_col = np.ones(self.nb_actions) / self.nb_actions
        else:
            sigma_row, sigma_col = sol

        sigma_row = safe_probvec(sigma_row, self.nb_actions)
        sigma_col = safe_probvec(sigma_col, self.nb_actions)

        # store predicted opponent strategy if you want diagnostics
        self.last_sigma_col_pred = sigma_col

        return int(self.rng.choice(self.nb_actions, p=sigma_row))

    def observe_opponent_move(self, m: int) -> None:
        self.m = int(m)

    def update_model(self) -> float:
        assert self.A is not None and self.B is not None

        # ---------- (a) error estimate ----------
        # predict opponent move prob under our current estimates (as in pick_move)
        A_mod, B_mod = make_modified_game(self.A, self.B, self.last_att_agent_used, self.last_att_opp_used)
        sol = solve_robust(nash.Game(A_mod, B_mod), self.last_nash_opp)
        if sol is None:
            sigma_col = np.ones(self.nb_actions) / self.nb_actions
        else:
            _, sigma_col = sol
        sigma_col = safe_probvec(sigma_col, self.nb_actions)

        p_obs = float(sigma_col[self.m])

        att_est = self.clip_att(self.estimate_attitude())
        bel_est = self.clip_att(self.estimate_belief())
        coop = float(np.clip(self.cooperation(att_est, bel_est), -1.0, 1.0))

        if self.use_lookup:
            j = self.bin_index(p_obs, self.prob_edges)
            k = self.bin_index(coop, self.coop_edges)
            lik = np.asarray(self.T[j, k, :], dtype=float)

            self.P_error *= lik
            s = float(self.P_error.sum())
            if (not np.isfinite(s)) or s <= 0.0:
                self.P_error[:] = 1.0 / len(self.P_error)
            else:
                self.P_error /= s

            error_est = float(np.dot(self.P_error, self.error_levels))
        else:
            error_est = 0.2

        error_est = float(np.clip(error_est, 1e-6, 2.0))

        # ---------- (b) resample ----------
        weights = np.zeros(self.n, dtype=float)

        for i, p in enumerate(self.particles):
            # particle implies these used attitudes this round
            att_agent_used = self.clip_att(p.att + self.r)
            att_opp_used = self.clip_att(p.bel + self.r)

            pA, pB = make_modified_game(self.A, self.B, att_agent_used, att_opp_used)
            psol = solve_robust(nash.Game(pA, pB), p.nash)

            if psol is None:
                p_sigma_col = np.ones(self.nb_actions) / self.nb_actions
            else:
                _, p_sigma_col = psol

            p_sigma_col = safe_probvec(p_sigma_col, self.nb_actions)
            weights[i] = float(p_sigma_col[self.m])

        sw = float(weights.sum())
        if sw <= 0.0 or (not np.isfinite(sw)):
            weights[:] = 1.0 / self.n
        else:
            weights /= sw

        idx = self.rng.choice(self.n, size=self.n, replace=True, p=weights)
        resampled = [self.particles[i] for i in idx]

        # ---------- (c) perturb ----------
        sigma = float(max(1e-6, error_est * self.fab))
        new_particles: list[Particle] = []

        for p in resampled:
            att_new = self.clip_att(self.rng.normal(loc=p.att, scale=sigma))
            bel_new = self.clip_att(self.rng.normal(loc=p.bel, scale=sigma))
            nash_new = int(p.nash)

            if self.rng.random() < float(np.clip(error_est * self.fnash, 0.0, 1.0)):
                nash_new = int(self.rng.integers(0, 2 * self.nb_actions))

            new_particles.append(Particle(att=att_new, bel=bel_new, nash=nash_new))

        self.particles = new_particles
        return error_est
