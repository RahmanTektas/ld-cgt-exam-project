from __future__ import annotations

import os
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from modified_game import make_modified_game

from game import random_bimatrix_game
from CooperativeAgent import CooperativeAgentAlgorithm

A = np.random.default_rng(0).random((4,4))
B = np.random.default_rng(1).random((4,4))

A_mod, _ = make_modified_game(A, B, att_row=0.0, att_col=0.0)
print("att_row=0 -> A_mod==A:", np.allclose(A_mod, A))

A_mod, _ = make_modified_game(A, B, att_row=1.0, att_col=0.0)
print("att_row=+1 -> A_mod==A+B:", np.allclose(A_mod, A + B))

A_mod, _ = make_modified_game(A, B, att_row=-1.0, att_col=0.0)
print("att_row=-1 -> A_mod==A-B:", np.allclose(A_mod, A - B))
