# main.py (squelette minimal)
import numpy as np
from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm

def run(seed=0, T=1000, n_actions=16):
    rng = np.random.default_rng(seed)

    agent1 = CooperativeAgentAlgorithm(nb_actions=n_actions)
    agent2 = CooperativeAgentAlgorithm(nb_actions=n_actions)

    for t in range(T):
        A, B = random_bimatrix_game(n_actions, rng=rng)

        # IMPORTANT: les deux doivent observer le même jeu
        agent1.A, agent1.B = A, B
        agent2.A, agent2.B = A, B

        a1 = agent1.pick_move()
        a2 = agent2.pick_move()

        r1 = A[a1, a2]
        r2 = B[a1, a2]

        # chacun observe le move adverse, puis update
        agent1.m = a2
        agent2.m = a1
        agent1.update_model()
        agent2.update_model()

    print("done")

if __name__ == "__main__":
    run()
