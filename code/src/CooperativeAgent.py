from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from collections import Counter

from modified_game import make_modified_game
from nash import one_nash_equilibrium


@dataclass
class Particle:
    att: float  # attitude
    bel: float  # belief
    nash: int   # label of method to choose a Nash equilibrium


class CooperativeAgentAlgorithm:

    ####### 1. Initialize #######
    def __init__(
        self,
        ### (a) Select values for parameters ###
        rng: np.random.Generator,
        nb_actions: int = 16,                    # Number of actions   (16 is the best empirical number of actions given by authors)
        nb_particles: int = 100,                 # Number of particles (No given number in the paper)
        reciprocation: float = 0.1,              # Reciprocation level (Authors use a reciprocation level of .1)
        fab: float = 0.1,                        # Perturbation factor for attitude and belief (Authors use 10% of the error in the current estimate)
        fnash: float = 0.05,                     # Perturbation factor for methods of picking Nash equilibria (No given number in the paper)
        error_levels: np.ndarray | None = None,  # Set of error levels (No given number in the paper)
        p_error: np.ndarray | None = None,       # Distribution over error levels (No given number in the paper)
        lookup_path: str | None = None,          # Path to lookup_table.npz
    ):
        self.nb_actions = nb_actions
        self.n = nb_particles
        self.r = reciprocation
        self.fab = fab
        self.fnash = fnash
        self.rng = rng

        ### (b) lookup table T with t(j,k,l) ###
        # We load a precomputed lookup table:
        #   T[j,k,l] ≈ P( observed-probability-bin = j | cooperation-bin = k, error-level = l )
        # This lets us update P_error online.

        self.use_lookup = False

        if lookup_path is not None:
            data = np.load(lookup_path, allow_pickle=True)

            self.T = data["T"]  # shape (J, K, L)
            self.prob_edges = data["prob_edges"]  # length J+1
            self.coop_edges = data["coop_edges"]  # length K+1

            if error_levels is None:
                self.error_levels = data["error_levels"]  # length L
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

        ### (c) initial particle set ###
        set_particles = []

        for i in range(self.n):
            att = float(np.clip(self.rng.normal(0.0, 1.0), -1.0, 1.0))
            bel = float(np.clip(self.rng.normal(0.0, 1.0), -1.0, 1.0))
            nash = int(self.rng.integers(0, 2 * self.nb_actions))  # uniform over possible dropped labels
            set_particles.append(Particle(att=att, bel=bel, nash=nash))

        self.particles = set_particles

        # keeps last nash_opp for pick and update
        self.last_nash_opp = 0
        self.last_att_opp = 0.0
        self.last_bel_opp = 0.0

        # observed opponent move
        self.m = 0

    ####### 2. Observe game G #######

    def observe_game(self, A: np.ndarray, B: np.ndarray):
        self.A = A
        self.B = B

    ####### Helper functions #######

    @staticmethod
    def cooperation(att: float, bel: float) -> float:
        # Paper's cooperation score (Eq.3 in your builder script)
        denom = np.sqrt(att * att + 1.0) * np.sqrt(bel * bel + 1.0)
        return float((att + bel) / denom)

    @staticmethod
    def bin_index(x: float, edges: np.ndarray) -> int:
        idx = np.searchsorted(edges, x, side="right") - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    @staticmethod
    def safe_probvec(p: np.ndarray, n: int) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if (not np.isfinite(p).all()) or p.sum() <= 0.0:
            return np.ones(n, dtype=float) / n
        p = np.clip(p, 0.0, None)
        s = p.sum()
        return (p / s) if s > 0.0 else (np.ones(n, dtype=float) / n)

    ####### 3. Pick Move #######

    def pick_move(self):
        # (a) Estimate opponent’s parameters
        att_opp = self.estimate_attitude()
        bel_opp = self.estimate_belief()
        nash_opp = Counter([p.nash for p in self.particles]).most_common(1)[0][0]  # most frequent Pnash

        self.last_att_opp = att_opp
        self.last_bel_opp = bel_opp
        self.last_nash_opp = int(nash_opp)

        # (b) Set attitude att_agent = att_opp + r
        att_agent = float(np.clip(att_opp + self.r, -1.0, 1.0))

        # (c) Construct modified game G'
        A_mod, B_mod = make_modified_game(self.A, self.B, att_agent, att_opp)
        sigma_row, sigma_col = one_nash_equilibrium(A_mod, B_mod, int(nash_opp))
        self.last_sigma_col_pred = sigma_col

        # (d) Draw move from ne_agent
        sigma_row = self.safe_probvec(sigma_row, self.nb_actions)
        return int(self.rng.choice(self.nb_actions, p=sigma_row))

    ####### 4. Observe opponent move m #######

    def observe_opponent_move(self, m: int) -> None:
        self.m = int(m)

    def estimate_attitude(self) -> float:
        return float(np.mean([p.att for p in self.particles]))

    def estimate_belief(self) -> float:
        return float(np.mean([p.bel for p in self.particles]))

    def estimate_nash(self) -> int:
        return int(Counter([p.nash for p in self.particles]).most_common(1)[0][0])

    ####### 5. Update Model #######

    def update_model(self):
        # (a) Update error estimate

        # i. Set attitude att_agent = bel_opp
        att_agent = float(self.last_bel_opp)

        # ii. Construct modified game G' and find its Nash equilibrium ne using nash_opp
        att_opp = float(self.last_att_opp)
        A_mod, B_mod = make_modified_game(self.A, self.B, att_agent, att_opp)
        _, sigma_col = one_nash_equilibrium(A_mod, B_mod, int(self.last_nash_opp))
        sigma_col = self.safe_probvec(sigma_col, self.nb_actions)

        # iii. Set j = probability assigned by ne_opp to observed opponent move
        p_obs = float(sigma_col[self.m])

        # iv. Calculate cooperation value k of estimated attitude and belief
        # (we use our current estimated opponent parameters from particles)
        att_est = float(np.clip(self.estimate_attitude(), -1.0, 1.0))
        bel_est = float(np.clip(self.estimate_belief(), -1.0, 1.0))
        coop = float(np.clip(self.cooperation(att_est, bel_est), -1.0, 1.0))

        # v. Update the probability of each error level l
        # vi. Normalize the distribution over error levels
        # vii. Estimate current level of error

        if self.use_lookup:
            j = self.bin_index(p_obs, self.prob_edges)
            k = self.bin_index(coop, self.coop_edges)

            # likelihood of seeing bin j given (k, l)
            lik = np.asarray(self.T[j, k, :], dtype=float)  # shape (L,)

            self.P_error *= lik
            s = float(self.P_error.sum())
            if (not np.isfinite(s)) or s <= 0.0:
                self.P_error[:] = 1.0 / len(self.P_error)
            else:
                self.P_error /= s

            error_est = float(np.dot(self.P_error, self.error_levels))
        else:
            # fallback (no lookup table loaded)
            error_est = 0.2

        # (b) Resample particles

        # i. Calculate the weight for each particle
        weights = np.zeros(self.n, dtype=float)
        for i, p in enumerate(self.particles):
            # A. Create modified game using p_att_i and p_bel_i and calculate its ne using p_nash_i
            p_A, p_B = make_modified_game(self.A, self.B, att_row=p.bel, att_col=p.att)
            _, p_sigma_col = one_nash_equilibrium(p_A, p_B, int(p.nash))
            p_sigma_col = self.safe_probvec(p_sigma_col, self.nb_actions)

            # B. Set weight for particle pi to nemop
            weights[i] = float(p_sigma_col[self.m])

        # ii. Draw n particles from the current set of particles using the calculated weights
        if weights.sum() == 0.0:
            weights[:] = 1.0
        weights = weights / weights.sum()

        p_idx = self.rng.choice(self.n, size=self.n, p=weights)
        new_particles = [self.particles[i] for i in p_idx]

        # (c) Perturb particles
        perturb_particles = []
        sigma = max(1e-6, error_est * self.fab)  # stable, simple

        for p in new_particles:
            # i. Modify attitude of each particle
            att_new = float(np.clip(self.rng.normal(loc=p.att, scale=sigma), -1.0, 1.0))

            # ii. Modify belief of each particle
            bel_new = float(np.clip(self.rng.normal(loc=p.bel, scale=sigma), -1.0, 1.0))

            # iii. With probability err * fnash draw a new method of calculating Nash equilibria for each particle
            nash_new = int(p.nash)
            if self.rng.random() < error_est * self.fnash:
                nash_new = int(self.rng.integers(0, 2 * self.nb_actions))

            perturb_particles.append(Particle(att=att_new, bel=bel_new, nash=nash_new))

        self.particles = perturb_particles
        return error_est
