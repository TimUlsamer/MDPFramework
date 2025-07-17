# mdp_framework/core/discrete.py

import numpy as np
from .base import BaseMDP

class DiscreteMDP(BaseMDP):
    """
    Diskretes MDP mit endlicher Zustands- und Aktionsmenge.
    """

    def __init__(self, n_states, n_actions, P, R, gamma):
        """
        P: Übergangswahrscheinlichkeiten (n_states, n_actions, n_states)
        R: Rewards (n_states, n_actions)
        gamma: Diskontierungsfaktor
        """
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.P = np.array(P)  # shape: (n_states, n_actions, n_states)
        self.R = np.array(R)  # shape: (n_states, n_actions)
        self.gamma = float(gamma)
        self.state = self.reset()

    def reset(self):
        """
        Setzt das MDP auf einen zufälligen Startzustand zurück.
        """
        self.state = np.random.randint(self.n_states)
        return self.state

    def step(self, action):
        """
        Führt die Aktion aus, gibt (nächster Zustand, Reward) zurück.
        """
        assert 0 <= action < self.n_actions
        probs = self.P[self.state, action]
        next_state = np.random.choice(self.n_states, p=probs)
        reward = self.R[self.state, action]
        self.state = next_state
        return next_state, reward

    def sample_state(self):
        """
        Gibt einen zufälligen Zustand zurück.
        """
        return np.random.randint(self.n_states)

    def sample_action(self):
        """
        Gibt eine zufällige Aktion zurück.
        """
        return np.random.randint(self.n_actions)

    def to_dict(self):
        """
        Gibt eine serialisierbare Repräsentation zurück.
        """
        return {
            "type": "discrete",
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "P": self.P.tolist(),
            "R": self.R.tolist(),
            "gamma": self.gamma
        }

    @classmethod
    def from_dict(cls, data):
        """
        Erzeugt eine MDP-Instanz aus serialisierten Daten.
        """
        return cls(
            n_states=data["n_states"],
            n_actions=data["n_actions"],
            P=np.array(data["P"]),
            R=np.array(data["R"]),
            gamma=data["gamma"]
        )
