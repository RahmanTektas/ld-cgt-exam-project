from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from collections import Counter

from modified_game import make_modified_game
from nash import one_nash_equilibrium
from metrics import cooperation
from game import random_bimatrix_game


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
        nb_actions: int = 16,                    # Number of actions   (16 is the best empirical number of actions given by authors)
        nb_particles: int = 500,                 # Number of particles (No given number in the paper)
        reciprocation: float = 0.1,              # Reciprocation level (Authors use a reciprocation level of .1)
        fab: float = 0.1,                        # Perturbation factor for attitude and belief (Authors use 10% of the error in the current estimate)
        fnash: float = 0.05,                     # Perturbation factor for methods of picking Nash equilibria (No given number in the paper)
        error_levels: np.ndarray | None = None,  # Set of error levels (No given number in the paper)
        p_error: np.ndarray | None = None,       # Distribution over error levels (No given number in the paper)
    ):
        self.nb_actions = nb_actions
        self.n = nb_particles
        self.r = reciprocation
        self.fab = fab
        self.fnash = fnash


        ### (b) lookup table T with t(j,k,l) ###

        # TODO

        ### (c) initial particle set ###
        set = []
        for i in range(self.n):
            rng = np.random.default_rng(seed=42)
            att = rng.normal(0.0, 1.0)
            bel = rng.normal(0.0, 1.0)
            nash = np.random.Generator.integers(0, 2 * self.nb_actions) # drawn from a uniform distribution over the set of possible starting parameters of L-H algorithm
            set.append(Particle(att=att, bel=bel, nash=nash))
        self.particles = set

        # keeps last nash_opp for pick and update
        self.last_nash_opp = 0
        


    ####### 2. Observe game G #######

    def observe_game(self):
        self.A, self.B = random_bimatrix_game(self.n)

    ####### 3. Pick Move #######

    def pick_move(self):
        # (a) Estimate opponent’s parameters
        att_opp = np.mean([p.att for p in self.particles])
        bel_opp = np.mean([p.bel for p in self.particles])
        nash_opp = Counter([p.nash for p in self.particles]).most_common(1)[0][0] # from the most frequent value of Pnash

        self.last_att_opp = att_opp
        self.last_bel_opp = bel_opp
        self.last_nash_opp = nash_opp

        # (b) Set attitude att_agent = att_opp + r
        att_agent = min(1.0, att_opp + self.r)
        att_agent = max(0.0, att_agent)

        # (c) Construct modified game G'
        
        A_mod, B_mod = make_modified_game(self.A, self.B, att_agent, att_opp)
        sigma_row, sigma_col = one_nash_equilibrium(A_mod, B_mod, nash_opp)

        # (d) Draw move from ne_agent
        return np.random.Generator.choice(self.nb_actions, p=sigma_row)

    ####### 4. Observe opponent move m #######
    
    def Observe_opponent(self):
        # TODO
        self.m = 3

     ####### 5. Update Model #######

    def update_model(self):
        # (a) Update error estimate 

        # i. Set attitude att_agent = bel_opp
        att_agent = self.last_bel_opp

        # ii. Construct modified game G' and find its Nash equilibrium ne using nash_opp
        att_opp = self.last_att_opp
        A_mod, B_mod = make_modified_game(self.A, self.B, att_agent, att_opp)
        sigma_row, sigma_col = one_nash_equilibrium(A_mod, B_mod, self.last_nash_opp)

        # iii. Set j = probability assigned by ne_opp to observed opponent move

        # TODO        

        # iv. Calculate cooperation value k of estimated attitude and belief
        
        # TODO

        # v. Update the probability of each error level l
        
        # TODO

        # vi. Normalize the distribution over error levels

        # TODO

        # vii. Estimate current level of error

        # TODO
        error_est = 0.2

        # (b) Resample particles

        # i. Calculate the weight for each particle
        weights = np.zeros(self.n, dtype=float)
        for i, p in enumerate(self.particles):

            # A. Create modified game using p_att_i and p_bel_i and calculate its ne using p_nash_i
            p_A, p_A = make_modified_game(self.A, self.B, att_row=p.bel, att_col=p.att)
            _, p_sigma_col = one_nash_equilibrium(p_A, p_A, p.nash)

            # B. Set weight for particle pi to nemop
            weights[i] = float(p_sigma_col[self.m])

        # ii. Draw n particles from the current set of particles using the calculated weights
        if weights.sum() == 0.0: # check for degenerate case
            weights[:] = 1.0
        weights = weights / weights.sum()

        p_idx = np.random.Generator.choice(self.n, size=self.n, p=weights)
        new_particles = [self.particles[i] for i in p_idx]

        # (c) Perturb particles 
        perturb_particles = []
        for p in new_particles:

            # i. Modify attitude of each particle
            att_new = np.random.Generator.normal(loc=p.att, scale=np.sqrt(max(0.0, error_est * self.fab)))

            # ii. Modify belief of each particlex
            bel_new = np.random.Generator.normal(loc=p.bel, scale=np.sqrt(max(0.0, error_est * self.fab)))

            # iii. With probability err ∗ fnash draw a new method of calculating Nash equilibria for each particle
            nash_new = p.nash
            if np.random.random(0.0, 1.0) < error_est * self.fnash:
                nash_new = np.random.Generator.integers(0, 2 * self.nb_actions)

            perturb_particles.append(Particle(att=att_new, bel=bel_new, nash=nash_new))

        self.particles = perturb_particles
        return error_est