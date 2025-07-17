# mdp_framework/generators/continuous_generator.py

import numpy as np
from mdp_framework.core.continuous import ContinuousMDP

class RandomLinearDynamics:
    def __init__(self, state_dim, action_dim, noise_std=0.01):
        self.A = np.random.randn(state_dim, state_dim)
        self.B = np.random.randn(state_dim, action_dim)
        self.noise_std = noise_std
        self.state_dim = state_dim

    def __call__(self, state, action):
        state = np.asarray(state)
        action = np.asarray(action)
        noise = np.random.randn(self.state_dim) * self.noise_std
        return self.A @ state + self.B @ action + noise

class RandomRewardFunction:
    def __init__(self, state_dim, action_dim):
        self.w_s = np.random.randn(state_dim)
        self.w_a = np.random.randn(action_dim)

    def __call__(self, state, action):
        state = np.asarray(state)
        action = np.asarray(action)
        return float(self.w_s @ state + self.w_a @ action)

def random_continuous_mdp(
        state_dim,
        action_dim,
        gamma=0.99,
        noise_std=0.01):
    dynamics_func = RandomLinearDynamics(state_dim, action_dim, noise_std=noise_std)
    reward_func = RandomRewardFunction(state_dim, action_dim)
    mdp = ContinuousMDP(state_dim, action_dim, dynamics_func, reward_func, gamma)
    return mdp

# Mini-Test
if __name__ == "__main__":
    mdp = random_continuous_mdp(4, 2)
    print("Initial state:", mdp.reset())
    for _ in range(5):
        a = mdp.sample_action()
        s_next, r = mdp.step(a)
        print(f"Action: {a}, Next state: {s_next}, Reward: {r}")
