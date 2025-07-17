#!/usr/bin/env python3

import argparse
import os
from mdp_framework.io.json_io import load_mdp_from_json, save_mdp_to_json
from mdp_framework.io.pickle_io import load_mdp_from_pickle, save_mdp_to_pickle
from mdp_framework.core.discrete import DiscreteMDP
from mdp_framework.core.continuous import ContinuousMDP

def main():
    parser = argparse.ArgumentParser(
        description="Konvertiert MDP-Dateien zwischen Pickle und JSON (nur diskrete MDPs)."
    )
    parser.add_argument("input", type=str, help="Eingabedatei (.json oder .pkl)")
    parser.add_argument(
        "--to",
        choices=["json", "pickle"],
        required=True,
        help="Zielformat (json oder pickle)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Pfad zur Ausgabedatei (optional)"
    )
    args = parser.parse_args()

    in_path = args.input
    to_format = args.to
    out_path = args.output

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {in_path}")

    # Bestimme Typ der Eingabedatei
    _, ext = os.path.splitext(in_path)
    ext = ext.lower()

    # Laden
    if ext == ".json":
        mdp = load_mdp_from_json(in_path)
    elif ext == ".pkl":
        mdp = load_mdp_from_pickle(in_path)
    else:
        raise ValueError("Eingabedatei muss .json oder .pkl sein.")

    # Typprüfung/Fehlermeldung
    if to_format == "json":
        if not isinstance(mdp, DiscreteMDP):
            raise TypeError("Nur diskrete MDPs können als JSON gespeichert werden.")
        out_path = out_path or os.path.splitext(in_path)[0] + ".json"
        save_mdp_to_json(mdp, out_path)
        print(f"MDP als JSON gespeichert: {out_path}")

    elif to_format == "pickle":
        out_path = out_path or os.path.splitext(in_path)[0] + ".pkl"
        save_mdp_to_pickle(mdp, out_path)
        print(f"MDP als Pickle gespeichert: {out_path}")

if __name__ == "__main__":
    main()
