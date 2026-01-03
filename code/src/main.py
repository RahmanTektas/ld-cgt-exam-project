import numpy as np
from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm

def run(seed=0, T=1000, n_actions=16):
    rng_env = np.random.default_rng(seed)
    rng1 =    np.random.default_rng(seed+1)
    rng2 =    np.random.default_rng(seed+2)

    agent1 = CooperativeAgentAlgorithm(nb_actions=n_actions, rng = rng1)
    agent2 = CooperativeAgentAlgorithm(nb_actions=n_actions, rng = rng2)

    for t in range(T):
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # both players observe the same game
        agent1.A, agent1.B = A, B
        agent2.A, agent2.B = A, B

        a1 = agent1.pick_move()
        a2 = agent2.pick_move()

        r1 = A[a1, a2]
        r2 = B[a1, a2]

        # each observe opponent move and update
        agent1.m = a2
        agent2.m = a1
        agent1.update_model()
        agent2.update_model()

    print("done")

if __name__ == "__main__":
    run()
