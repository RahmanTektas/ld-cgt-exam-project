# Cooperative-Agent Game Theory Reproduction

Course project for **Learning Dynamics / Computational Game Theory** at ULB/VUB.

The repository contains experimental code for reproducing a cooperative-agent model in repeated bimatrix games. The implementation estimates an opponent's behavioural parameters with a particle-based model and uses Nash-equilibrium calculations to choose actions.

## Main components

- particle representation of opponent attitude, belief, and equilibrium-selection state
- repeated random bimatrix-game simulation
- robust Lemke–Howson equilibrium computation with Nashpy
- particle reweighting and resampling from observed opponent actions
- lookup-table based error estimation
- experiment logging and plotting scripts
- repeated-seed result files for comparison and analysis

## Code layout

```text
code/src/
  CooperativeAgent.py   particle-based opponent model and action selection
  nash.py               equilibrium solver wrapper
  modified_game.py      payoff transformation
  main.py               simulation runner
  metrics.py            evaluation metrics
  plots/                 reproduction figures

report/                  course report sources
```

## Status

This is an academic reproduction repository rather than a packaged library. The code and experiment artifacts are kept primarily to document the implementation and analysis used for the course project.


## Environment

The implementation uses Python with NumPy, Nashpy, Matplotlib, and tqdm.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main experiment code lives under `code/src/`. The repository also keeps selected generated result files used for analysis and figure reproduction.
