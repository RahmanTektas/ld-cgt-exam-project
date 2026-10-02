import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "src"))

from modified_game import make_modified_game


def sample_game():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    return rng_a.random((4, 4)), rng_b.random((4, 4))


def test_zero_attitude_keeps_row_payoffs():
    a, b = sample_game()
    a_modified, _ = make_modified_game(a, b, att_row=0.0, att_col=0.0)
    assert np.allclose(a_modified, a)


def test_positive_attitude_adds_other_player_payoff():
    a, b = sample_game()
    a_modified, _ = make_modified_game(a, b, att_row=1.0, att_col=0.0)
    assert np.allclose(a_modified, a + b)


def test_negative_attitude_subtracts_other_player_payoff():
    a, b = sample_game()
    a_modified, _ = make_modified_game(a, b, att_row=-1.0, att_col=0.0)
    assert np.allclose(a_modified, a - b)
