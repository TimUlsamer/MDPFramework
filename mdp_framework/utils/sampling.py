# mdp_framework/utils/sampling.py

import numpy as np

def sample_discrete(probabilities):
    """
    Zieht einen Index entsprechend des gegebenen diskreten Wahrscheinlichkeitsvektors.
    """
    probabilities = np.asarray(probabilities)
    if not np.isclose(probabilities.sum(), 1):
        raise ValueError("Wahrscheinlichkeiten müssen auf 1 normiert sein.")
    idx = np.random.choice(len(probabilities), p=probabilities)
    return idx

def sample_uniform(low, high, shape=None):
    """
    Einfaches uniformes Sampling (wrapper für np.random.uniform).
    """
    return np.random.uniform(low, high, size=shape)

# Mini-Test
if __name__ == "__main__":
    print("Diskretes Sample:", sample_discrete([0.1, 0.3, 0.6]))
    print("Uniformes Sample:", sample_uniform(-5, 5, (3,)))
