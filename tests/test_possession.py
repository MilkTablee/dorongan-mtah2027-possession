"""Fast unit tests for the decoder and the classifier's inputs. No match data needed.

    pip install pytest
    python -m pytest tests
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.data import Match  # noqa: E402
from possession.learned import table  # noqa: E402
from possession.model import (A, D, H, MIN_DEAD_FRAMES, absorb_short_dead,  # noqa: E402
                              features, viterbi)

COSTS = {"swap": 10, "stop": 24, "start": 24}


def test_viterbi_holds_state_under_weak_contrary_evidence():
    score = np.zeros((200, 3))
    score[:, H] = 1.0
    score[100:105, A] = 2.0  # five frames of weak evidence: cheaper to stay than to swap twice
    assert set(viterbi(score, COSTS)) == {H}


def test_viterbi_swaps_on_sustained_evidence():
    score = np.zeros((400, 3))
    score[:200, H] = 1.0
    score[200:, A] = 1.0
    path = viterbi(score, COSTS)
    assert path[0] == H and path[-1] == A
    assert int(np.sum(np.diff(path) != 0)) == 1


def test_short_dead_spells_go_to_the_team_before():
    path = np.full(300, H, dtype=np.int8)
    path[100:100 + MIN_DEAD_FRAMES - 1] = D
    assert set(absorb_short_dead(path)) == {H}


def test_real_stoppages_survive():
    path = np.full(300, H, dtype=np.int8)
    path[100:100 + MIN_DEAD_FRAMES] = D
    assert (absorb_short_dead(path) == path).all()


def _random_match(seed, n=600, slots=22):
    rng = np.random.default_rng(seed)
    players = np.cumsum(rng.normal(0, 0.2, (n, slots, 2)), axis=0) + [52.5, 34.0]
    ball = np.cumsum(rng.normal(0, 0.5, (n, 2)), axis=0) + [52.5, 34.0]
    ball[rng.random(n) < 0.4] = np.nan
    team = np.array([0] * (slots // 2) + [1] * (slots // 2))
    return Match("synthetic", ball.astype(np.float32), players.astype(np.float32), team,
                 [(0, n - 1)], None, None)


def test_swapping_the_teams_mirrors_the_features():
    """The model is trained on both orientations, so a swap must give exactly the mirrored table."""
    match = _random_match(0)
    swapped = match._replace(team=1 - match.team)
    straight, mirrored = table(match, features(match, 0))
    swapped_straight, _ = table(swapped, features(swapped, 0))
    np.testing.assert_allclose(swapped_straight, mirrored, rtol=1e-5, atol=1e-5, equal_nan=True)
