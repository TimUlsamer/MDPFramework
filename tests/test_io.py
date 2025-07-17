# tests/test_io.py

import os
import tempfile
from mdp_framework.generators.discrete_generator import random_discrete_mdp
from mdp_framework.generators.continuous_generator import random_continuous_mdp
from mdp_framework.io.json_io import save_mdp_to_json, load_mdp_from_json
from mdp_framework.io.pickle_io import save_mdp_to_pickle, load_mdp_from_pickle

def test_json_save_load_discrete():
    mdp = random_discrete_mdp(4, 3)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        save_mdp_to_json(mdp, tmp.name)
        mdp_loaded = load_mdp_from_json(tmp.name)
    assert mdp_loaded.n_states == mdp.n_states
    assert mdp_loaded.n_actions == mdp.n_actions

def test_pickle_save_load_discrete():
    mdp = random_discrete_mdp(3, 2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        save_mdp_to_pickle(mdp, tmp.name)
        mdp_loaded = load_mdp_from_pickle(tmp.name)
    assert mdp_loaded.n_states == mdp.n_states

def test_pickle_save_load_continuous():
    mdp = random_continuous_mdp(3, 2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        save_mdp_to_pickle(mdp, tmp.name)
        mdp_loaded = load_mdp_from_pickle(tmp.name)
    assert mdp_loaded.state_dim == mdp.state_dim
    assert mdp_loaded.action_dim == mdp.action_dim

