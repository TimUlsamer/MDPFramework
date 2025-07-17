# mdp_framework/utils/validators.py

import numpy as np

def is_stochastic_matrix(P):
    """
    Prüft, ob P eine stochastische Matrix ist (jede Zeile summiert auf 1).
    """
    P = np.asarray(P)
    if P.ndim != 2:
        return False
    return np.allclose(P.sum(axis=1), 1)

def validate_discrete_mdp_shapes(P, R, n_states, n_actions):
    """
    Prüft, ob P und R die erwarteten Shapes für ein diskretes MDP haben.
    """
    P = np.asarray(P)
    R = np.asarray(R)
    if P.shape != (n_states, n_actions, n_states):
        raise ValueError(f"P muss Shape {(n_states, n_actions, n_states)} haben, hat aber {P.shape}")
    if R.shape != (n_states, n_actions):
        raise ValueError(f"R muss Shape {(n_states, n_actions)} haben, hat aber {R.shape}")

# Mini-Test
if __name__ == "__main__":
    P = np.ones((2, 3, 2)) / 2
    R = np.zeros((2, 3))
    print("Ist stochastisch:", is_stochastic_matrix(P[0, 0]))
    validate_discrete_mdp_shapes(P, R, 2, 3)
    print("Validation erfolgreich.")
