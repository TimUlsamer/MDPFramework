# mdp_framework/io/pickle_io.py

import pickle
import os
from mdp_framework.core.discrete import DiscreteMDP
from mdp_framework.core.continuous import ContinuousMDP

def save_mdp_to_pickle(mdp, filepath):
    """
    Speichert beliebige MDP-Objekte (diskret oder stetig) als Pickle-Datei.
    ACHTUNG: Pickle-Dateien sind nicht portabel zwischen Python-Versionen
    und potentiell UNSICHER (niemals von untrusted sources laden!).
    """
    with open(filepath, "wb") as f:
        pickle.dump(mdp, f)

def load_mdp_from_pickle(filepath):
    """
    Lädt beliebige MDP-Objekte (diskret oder stetig) aus einer Pickle-Datei.
    """
    with open(filepath, "rb") as f:
        mdp = pickle.load(f)
    if not isinstance(mdp, (DiscreteMDP, ContinuousMDP)):
        raise TypeError("Geladenes Objekt ist kein gültiges MDP.")
    return mdp

# Mini-Test
if __name__ == "__main__":
    from mdp_framework.generators.continuous_generator import random_continuous_mdp
    mdp = random_continuous_mdp(3, 2)
    save_path = "test_mdp.pkl"
    save_mdp_to_pickle(mdp, save_path)
    print(f"MDP als {save_path} gespeichert.")

    loaded_mdp = load_mdp_from_pickle(save_path)
    print("Geladenes MDP, erster Zustand:", loaded_mdp.reset())
    os.remove(save_path)
