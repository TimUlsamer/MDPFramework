# mdp_framework/io/json_io.py

import json
import os
from mdp_framework.core.discrete import DiscreteMDP
# ContinuousMDP wird importiert, aber nur für Typprüfung und Error-Handling
from mdp_framework.core.continuous import ContinuousMDP

def save_mdp_to_json(mdp, filepath):
    """
    Speichert ein diskretes MDP als JSON-Datei.
    Für stetige MDPs wird ein Fehler ausgelöst (da Funktionen nicht serialisierbar).
    """
    if isinstance(mdp, DiscreteMDP):
        data = mdp.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    elif isinstance(mdp, ContinuousMDP):
        raise NotImplementedError("ContinuousMDP kann nicht als JSON gespeichert werden (Funktionen sind nicht serialisierbar).")
    else:
        raise TypeError("Unbekannter MDP-Typ.")

def load_mdp_from_json(filepath):
    """
    Lädt ein diskretes MDP aus einer JSON-Datei.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    if data.get("type") == "discrete":
        return DiscreteMDP.from_dict(data)
    elif data.get("type") == "continuous":
        raise NotImplementedError("ContinuousMDP kann nicht aus JSON geladen werden (Funktionen fehlen).")
    else:
        raise ValueError("Unbekannter oder nicht unterstützter MDP-Typ in Datei.")

# Mini-Test
if __name__ == "__main__":
    from mdp_framework.generators.discrete_generator import random_discrete_mdp
    mdp = random_discrete_mdp(4, 2)
    save_path = "test_mdp.json"
    save_mdp_to_json(mdp, save_path)
    print(f"MDP als {save_path} gespeichert.")

    loaded_mdp = load_mdp_from_json(save_path)
    print("Geladenes MDP, erster Zustand:", loaded_mdp.reset())
    os.remove(save_path)
