#!/usr/bin/env python3

import argparse
import os
from mdp_framework.config import (
    DISCRETE_DATA_PATH,
    CONTINUOUS_DATA_PATH,
    DEFAULT_N_STATES,
    DEFAULT_N_ACTIONS,
    DEFAULT_STATE_DIM,
    DEFAULT_ACTION_DIM,
    DEFAULT_GAMMA,
    set_global_seed,
)
from mdp_framework.generators.discrete_generator import random_discrete_mdp
from mdp_framework.generators.continuous_generator import random_continuous_mdp
from mdp_framework.io.json_io import save_mdp_to_json
from mdp_framework.io.pickle_io import save_mdp_to_pickle

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description="Erzeugt zufällige MDPs (diskret oder stetig) und speichert sie ab."
    )
    parser.add_argument("--type", choices=["discrete", "continuous"], required=True, help="MDP-Typ")
    parser.add_argument("--n_states", type=int, default=DEFAULT_N_STATES, help="Anzahl Zustände (diskret)")
    parser.add_argument("--n_actions", type=int, default=DEFAULT_N_ACTIONS, help="Anzahl Aktionen (diskret)")
    parser.add_argument("--state_dim", type=int, default=DEFAULT_STATE_DIM, help="Dimensionalität des Zustands (stetig)")
    parser.add_argument("--action_dim", type=int, default=DEFAULT_ACTION_DIM, help="Dimensionalität der Aktion (stetig)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount-Faktor")
    parser.add_argument("--seed", type=int, default=None, help="Globaler Zufallsseed")
    parser.add_argument("--count", type=int, default=1, help="Wieviele MDPs erzeugen?")
    parser.add_argument("--name", type=str, default=None, help="Basisname für MDP-Datei(en)")
    parser.add_argument("--no_pickle", action="store_true", help="Diskrete MDPs NICHT zusätzlich als Pickle speichern")

    args = parser.parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)

    if args.type == "discrete":
        ensure_dir_exists(DISCRETE_DATA_PATH)
        for i in range(args.count):
            mdp = random_discrete_mdp(args.n_states, args.n_actions, gamma=args.gamma)
            base_filename = args.name or f"discrete_{args.n_states}_{args.n_actions}_mdp_{i+1}"
            json_path = os.path.join(DISCRETE_DATA_PATH, f"{base_filename}.json")
            save_mdp_to_json(mdp, json_path)
            print(f"Diskretes MDP gespeichert: {json_path}")
            if not args.no_pickle:
                pkl_path = os.path.join(DISCRETE_DATA_PATH, f"{base_filename}.pkl")
                save_mdp_to_pickle(mdp, pkl_path)
                print(f"Auch als Pickle gespeichert: {pkl_path}")

    elif args.type == "continuous":
        ensure_dir_exists(CONTINUOUS_DATA_PATH)
        for i in range(args.count):
            mdp = random_continuous_mdp(args.state_dim, args.action_dim, gamma=args.gamma)
            base_filename = args.name or f"continuous_{args.state_dim}_{args.action_dim}_mdp_{i+1}"
            pkl_path = os.path.join(CONTINUOUS_DATA_PATH, f"{base_filename}.pkl")
            save_mdp_to_pickle(mdp, pkl_path)
            print(f"Stetiges MDP gespeichert: {pkl_path}")

if __name__ == "__main__":
    main()
