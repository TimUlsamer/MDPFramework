# mdp_framework/generators/continuous_generator.py

import numpy as np
from mdp_framework.core.continuous import ContinuousMDP

def random_linear_dynamics(state_dim, action_dim, noise_std=0.01):
    """
    Erstellt eine lineare Zufallsdynamikfunktion für ContinuousMDP.
    """
    A = np.random.randn(state_dim, state_dim)
    B = np.random.randn(state_dim, action_dim)

    def dynamics_func(state, action):
        state = np.asarray(state)
        action = np.asarray(action)
        noise = np.random.randn(state_dim) * noise_std
        return A @ state + B @ action + noise

    return dynamics_func

def random_reward_func(state_dim, action_dim):
    """
    Erstellt eine lineare Zufalls-Rewardfunktion für ContinuousMDP.
    """
    w_s = np.random.randn(state_dim)
    w_a = np.random.randn(action_dim)

    def reward_func(state, action):
        state = np.asarray(state)
        action = np.asarray(action)
        return float(w_s @ state + w_a @ action)

    return reward_func

def random_continuous_mdp(
    state_dim,
    action_dim,
    gamma=0.99,
    noise_std=0.01
):
    """
    Erstellt ein zufälliges stetiges MDP mit linearer Dynamik und linearer Rewardfunktion.
    """
    dynamics_func = random_linear_dynamics(state_dim, action_dim, noise_std=noise_std)
    reward_func = random_reward_func(state_dim, action_dim)
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
