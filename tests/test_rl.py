import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from rl import RLModel  # noqa: E402


def test_q_table_shape_matches_environment():
    model = RLModel()
    assert model.q_table.shape == (16, 4)


def test_terminal_update_uses_reward_only():
    model = RLModel()
    model.update_qtable(state=14, action=2, reward=1.0, next_state=15, done=True)
    assert model.q_table[14][2] == model.learning_rate * 1.0


def test_agent_learns_to_solve_the_lake():
    np.random.seed(0)
    model = RLModel()
    model.env.reset(seed=0)
    model.episodes = 3000
    model.train()
    assert model.evaluate(episodes=20) == 1.0
