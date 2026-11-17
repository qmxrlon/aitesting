# Q-Learning: FrozenLake

A tabular **Q-learning** agent that teaches itself to cross the frozen lake from
[Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/),
with no training data, only trial, error and reward.

```
S F F F        S = start, G = goal
F H F H        H = hole (episode ends, reward 0)
F F F H        F = frozen, walkable
H F F G        reaching G gives reward 1
```

## How it works

The agent keeps a table `Q[state][action]` with the expected future reward of each move
and improves it after every step with the Bellman update:

```
Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s',·) − Q(s,a))
```

- **α = 0.6** learning rate, **γ = 0.95** discount
- **ε-greedy exploration**: starts fully random (ε = 1.0) and decays to 0.01, so the
  agent explores first and exploits what it learned later
- Truncated episodes (step limit) still bootstrap from the next state; only real
  terminal states (hole or goal) cut off the return

After training, the greedy policy reaches the goal in 100 % of episodes:

```
↓ → ↓ ←
↓ H ↓ H
→ ↓ ↓ H
H → → G
```

## Run

```bash
pip install -r requirements.txt
python rl.py      # trains 10,000 episodes, then prints success rate and policy
pytest            # tests
```
