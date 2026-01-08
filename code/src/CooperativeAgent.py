from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
import nashpy as nash

from src.modified_game import make_modified_game
from src.nash import solve_robust, safe_probvec, EPS


@dataclass
class Particle:
    att: float  # opponent attitude
    bel: float  # opponent belief about OUR attitude
    nash: int   # dropped label (Nash-selection proxy)


class CooperativeAgentAlgorithm:
    """
    Implements the Cooperative Agent Algorithm from AAAI'08 (paper box).
    Key semantics (matches the algorithm screenshot):
      - att modifies utilities directly (Eq. 1)
      - bel is what opponent believes about OUR attitude
      - att_agent = att_opp + r (Pick Move)
      - Error update uses att_agent = bel_opp (Update Model step 5a-i)
      - Particle weighting uses modified game with (att_row = p.bel, att_col = p.att)
    """
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

        # lookup table
        self.use_lookup = False
        if lookup_path is not None:
            data = np.load(lookup_path, allow_pickle=True)
            self.T = np.asarray(data["T"], dtype=float)  # (J,K,L)
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

        # particles init (paper: N(0,1) clipped)
        self.particles: list[Particle] = []
        for _ in range(self.n):
            att = float(np.clip(self.rng.normal(0.0, 1.0), -1.0, 1.0))
            bel = float(np.clip(self.rng.normal(0.0, 1.0), -1.0, 1.0))
            nash_label = int(self.rng.integers(0, 2 * self.nb_actions))
            self.particles.append(Particle(att=att, bel=bel, nash=nash_label))

        # per-round state
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.m: int = 0

        # store last estimated opponent params (needed for update step)
        self.last_att_opp_est = 0.0
        self.last_bel_opp_est = 0.0
        self.last_nash_opp = 0

    # ---------- helpers ----------
    @staticmethod
    def cooperation(att: float, bel: float) -> float:
        # paper Eq.(3)
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

        # (3a) Estimate opponent parameters
        att_opp_est = self.clip_att(self.estimate_attitude())
        bel_opp_est = self.clip_att(self.estimate_belief())
        nash_opp = self.estimate_nash()

        self.last_att_opp_est = att_opp_est
        self.last_bel_opp_est = bel_opp_est
        self.last_nash_opp = int(nash_opp)

        # (3b) Reciprocation rule (paper)
        att_agent = self.clip_att(att_opp_est + self.r)

        # (3c) Agent's modified utility ONLY (Eq. 1)
        A_mod, B_mod = make_modified_game(self.A, self.B, att_row=att_agent, att_col=att_opp_est)

        sol = solve_robust(nash.Game(A_mod, B_mod), nash_opp)
        if sol is None:
            sigma_row = np.ones(self.nb_actions) / self.nb_actions
            sigma_col = np.ones(self.nb_actions) / self.nb_actions
        else:
            sigma_row, sigma_col = sol

        sigma_row = safe_probvec(sigma_row, self.nb_actions)
        sigma_col = safe_probvec(sigma_col, self.nb_actions)

        self.last_sigma_col_pred = sigma_col
        return int(self.rng.choice(self.nb_actions, p=sigma_row))


    def observe_opponent_move(self, m: int) -> None:
        self.m = int(m)

    def update_model(self) -> float:
        """
        Paper step 5:
        (a) Update error estimate using opponent's predicted game-view:
            set att_agent = bel_opp, att_opp = att_opp, solve with opponent Nash-label,
            j = prob assigned to observed move.
        (b) Resample particles using likelihood of observed move under each particle.
        (c) Perturb particles with N(., err * f_ab) and mutate Nash-label with prob err * f_nash.
        """
        assert self.A is not None and self.B is not None

        EPS = 1e-12

        # (5a) error estimate(paper: att_agent = bel_opp)
        A_pred, B_pred = make_modified_game(
            self.A, self.B,
            att_row=float(self.last_bel_opp_est),  # att_agent = bel_opp
            att_col=float(self.last_att_opp_est),  # att_opp   = att_opp
        )

        sol = solve_robust(nash.Game(A_pred, B_pred), int(self.last_nash_opp))
        if sol is None:
            sigma_col = np.ones(self.nb_actions, dtype=float) / self.nb_actions
        else:
            _, sigma_col = sol
            sigma_col = safe_probvec(sigma_col, self.nb_actions)

        p_obs = float(sigma_col[self.m])  # paper's j

        # cooperation computed from CURRENT particle-mean estimates (raw)
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

        # (5b) resample
        weights = np.zeros(self.n, dtype=float)

        for i, p in enumerate(self.particles):
            pA, pB = make_modified_game(self.A, self.B, att_row=float(p.bel), att_col=float(p.att))

            psol = solve_robust(nash.Game(pA, pB), int(p.nash))
            if psol is None:
                p_sigma_col = np.ones(self.nb_actions, dtype=float) / self.nb_actions
            else:
                _, p_sigma_col = psol
                p_sigma_col = safe_probvec(p_sigma_col, self.nb_actions)

            weights[i] = max(float(p_sigma_col[self.m]), EPS)

        sw = float(weights.sum())
        if sw <= 0.0 or (not np.isfinite(sw)):
            weights[:] = 1.0 / self.n
        else:
            weights /= sw

        idx = self.rng.choice(self.n, size=self.n, replace=True, p=weights)
        resampled = [self.particles[i] for i in idx]

        # (5c) perturb
        sigma = float(max(1e-6, error_est * self.fab))
        p_change_nash = float(np.clip(error_est * self.fnash, 0.0, 1.0))

        new_particles: list[Particle] = []
        for p in resampled:
            att_new = self.clip_att(self.rng.normal(loc=float(p.att), scale=sigma))
            bel_new = self.clip_att(self.rng.normal(loc=float(p.bel), scale=sigma))

            nash_new = int(p.nash)
            if self.rng.random() < p_change_nash:
                nash_new = int(self.rng.integers(0, 2 * self.nb_actions))

            new_particles.append(Particle(att=att_new, bel=bel_new, nash=nash_new))

        self.particles = new_particles
        return error_est
