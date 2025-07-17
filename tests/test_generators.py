# tests/test_generators.py

import numpy as np
from mdp_framework.generators.discrete_generator import random_discrete_mdp
from mdp_framework.generators.continuous_generator import random_continuous_mdp

def test_random_discrete_mdp_shapes():
    n_states, n_actions = 6, 4
    mdp = random_discrete_mdp(n_states, n_actions)
    assert mdp.P.shape == (n_states, n_actions, n_states)
    assert mdp.R.shape == (n_states, n_actions)

def test_random_continuous_mdp_shapes():
    state_dim, action_dim = 5, 2
    mdp = random_continuous_mdp(state_dim, action_dim)
    s = mdp.reset()
    a = mdp.sample_action()
    s_next, r = mdp.step(a)
    assert s.shape == (state_dim,)
    assert s_next.shape == (state_dim,)
    assert isinstance(r, float)

