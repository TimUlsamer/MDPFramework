# mdp_framework/generators/discrete_generator.py

import numpy as np
from mdp_framework.core.discrete import DiscreteMDP

def random_discrete_mdp(
    n_states,
    n_actions,
    gamma=0.99,
    reward_std=1.0,
    random_start_state=True
):
    """
    Erstellt ein zufälliges diskretes MDP.

    - n_states: Anzahl Zustände
    - n_actions: Anzahl Aktionen
    - gamma: Diskontierungsfaktor
    - reward_std: Standardabweichung für Rewards
    - random_start_state: Wenn True, wird bei reset ein zufälliger Startzustand gewählt
    """

    # Übergangswahrscheinlichkeiten P[s, a, s'] mit Dirichlet für Stochastizität
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            P[s, a] = np.random.dirichlet(np.ones(n_states))

    # Rewards (normalverteilt)
    R = np.random.normal(loc=0.0, scale=reward_std, size=(n_states, n_actions))

    mdp = DiscreteMDP(n_states, n_actions, P, R, gamma)
    if not random_start_state:
        mdp.state = 0
    return mdp

# Mini-Test
if __name__ == "__main__":
    mdp = random_discrete_mdp(5, 3)
    print("Initial state:", mdp.reset())
    for _ in range(5):
        a = mdp.sample_action()
        s_next, r = mdp.step(a)
        print(f"Action: {a}, Next state: {s_next}, Reward: {r}")
