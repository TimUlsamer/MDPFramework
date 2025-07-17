# mdp_framework/config.py

import os

# Standard-Speicherorte für MDP-Daten
DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
DISCRETE_DATA_PATH = os.path.join(DATA_ROOT, "discrete")
CONTINUOUS_DATA_PATH = os.path.join(DATA_ROOT, "continuous")

# Standard-Parameter für MDP-Erzeugung
DEFAULT_N_STATES = 5
DEFAULT_N_ACTIONS = 2
DEFAULT_STATE_DIM = 3
DEFAULT_ACTION_DIM = 2
DEFAULT_GAMMA = 0.99

# Globale Zufallsseed-Option (optional, für Reproduzierbarkeit)
GLOBAL_RANDOM_SEED = None

def set_global_seed(seed):
    """
    Setzt den globalen Zufallsseed für numpy.
    """
    global GLOBAL_RANDOM_SEED
    GLOBAL_RANDOM_SEED = seed
    import numpy as np
    np.random.seed(seed)

# Mini-Test
if __name__ == "__main__":
    print("DATA_ROOT:", DATA_ROOT)
    set_global_seed(42)
    import numpy as np
    print("Sample random:", np.random.rand())
