import numpy as np
from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm
from metrics import compute_metrics  
import argparse
from tqdm import tqdm


def run_stationary(T=1000, n_actions=16, seed=42):
    """Experiment: one learning agent vs a stationary opponent."""
    rng_env = np.random.default_rng(seed)
    rng_agent = np.random.default_rng(seed + 1)
    rng_opp = np.random.default_rng(seed + 2)

    # Opponent: fixed (stationary) 
    opp_att = rng_opp.normal(0.0, 1.0)
    opp_bel = rng_opp.normal(0.0, 1.0)
    opp_att = np.clip(opp_att, -1, 1)
    opp_bel = np.clip(opp_bel, -1, 1)
    print(f"[Opponent] attitude={opp_att:.2f}, belief={opp_bel:.2f}")

    # Agent
    agent = CooperativeAgentAlgorithm(nb_actions=n_actions, rng=rng_agent)

    # Logs 
    estimated_attitudes = []
    estimated_beliefs = []

    for t in tqdm(range(T)):
        # 1. new random game
        A, B = random_bimatrix_game(n_actions, rng=rng_env)

        # 2. Stationnary opponent picks a move
        opp_move = rng_opp.integers(0, n_actions)

        # 3. Agent picks a move
        agent.A, agent.B = A, B
        agent.m = opp_move
        agent_move = agent.pick_move()

        # 4. Payoffs
        r_agent = A[agent_move, opp_move]
        r_opp = B[agent_move, opp_move]

        # 5. Model update (particle filter)
        agent.update_model()

        # 6. Save metrics
        estimated_attitudes.append(agent.estimate_attitude())
        estimated_beliefs.append(agent.estimate_belief())

        if t % 100 == 0:
            print(f"[t={t}] Agent move={agent_move}, Opp move={opp_move}")

    compute_metrics(estimated_attitudes, estimated_beliefs)
    print("Simulation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["stationary"], default="stationary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=100) #1000
    parser.add_argument("--n_actions", type=int, default=4) #16
    args = parser.parse_args()

    if args.scenario == "stationary":
        run_stationary(T=args.T, n_actions=args.n_actions, seed=args.seed)
