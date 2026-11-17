import numpy as np
import gymnasium as gym

class RLModel:
    def __init__(self):
        self.learning_rate = 0.6
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.episodes = 10000
        self.env, self.n_states, self.actions = self.setup_env()
        self.q_table = np.zeros((self.n_states, self.actions))

    def setup_env(self, is_slippery: bool = False):
        self.env = gym.make(
            'FrozenLake-v1',
            desc=None,
            map_name='4x4',
            is_slippery=is_slippery,
        )
        return self.env, self.env.observation_space.n, self.env.action_space.n

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update_qtable(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def train(self):
        wins = 0
        for episode in range(self.episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _info = self.env.step(action)
                done = terminated or truncated
                if reward == 1.0:
                    wins += 1
                # Only a real terminal state ends the return; hitting the step
                # limit (truncation) should still bootstrap from the next state.
                self.update_qtable(state, action, reward, next_state, terminated)
                state = next_state

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % 200 == 0:
                win_rate = (wins / (episode + 1)) * 100
                print(f'Episode {episode} - Win rate: {win_rate:.2f}%')

    def evaluate(self, episodes: int = 100) -> float:
        """Play greedily (no exploration) and return the success rate (0–1)."""
        wins = 0
        for _ in range(episodes):
            state, _info = self.env.reset()
            done = False
            while not done:
                action = int(np.argmax(self.q_table[state]))
                state, reward, terminated, truncated, _info = self.env.step(action)
                done = terminated or truncated
            wins += int(reward == 1.0)
        return wins / episodes

    def policy_grid(self) -> str:
        """Render the learned policy as arrows on the 4x4 map."""
        arrows = ['←', '↓', '→', '↑']
        desc = self.env.unwrapped.desc.astype(str)
        rows = []
        for r, line in enumerate(desc):
            cells = []
            for c, tile in enumerate(line):
                state = r * len(line) + c
                cells.append(tile if tile in 'HG' else arrows[int(np.argmax(self.q_table[state]))])
            rows.append(' '.join(cells))
        return '\n'.join(rows)


if __name__ == '__main__':
    rl_model = RLModel()
    rl_model.train()
    print(f'\nGreedy success rate: {rl_model.evaluate() * 100:.0f}%')
    print(rl_model.policy_grid())
