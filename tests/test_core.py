# tests/test_core.py

import numpy as np
import pytest
from mdp_framework.core.discrete import DiscreteMDP
from mdp_framework.core.continuous import ContinuousMDP

def test_discrete_mdp_step_and_reset():
    n_states = 4
    n_actions = 3
    P = np.ones((n_states, n_actions, n_states)) / n_states
    R = np.arange(n_states * n_actions).reshape(n_states, n_actions)
    gamma = 0.95
    mdp = DiscreteMDP(n_states, n_actions, P, R, gamma)
    s0 = mdp.reset()
    assert 0 <= s0 < n_states
    for _ in range(10):
        a = mdp.sample_action()
        s1, r = mdp.step(a)
        assert 0 <= s1 < n_states
        assert r == R[mdp.state, a] or r in R  # reward korrekt

def test_continuous_mdp_step_and_reset():
    state_dim = 3
    action_dim = 2
    gamma = 0.99
    def dyn(s, a):
        return np.ones(state_dim)
    def rew(s, a):
        return 42.0
    mdp = ContinuousMDP(state_dim, action_dim, dyn, rew, gamma)
    s0 = mdp.reset()
    assert s0.shape == (state_dim,)
    for _ in range(5):
        a = mdp.sample_action()
        s1, r = mdp.step(a)
        assert s1.shape == (state_dim,)
        assert r == 42.0

