# mdp_framework/core/continuous.py

import numpy as np
from .base import BaseMDP

class ContinuousMDP(BaseMDP):
    """
    Stetiges MDP mit kontinuierlichem Zustands- und Aktionsraum.
    Die Dynamik und die Reward-Funktion werden als Funktionen übergeben.
    """

    def __init__(self, state_dim, action_dim, dynamics_func, reward_func, gamma):
        """
        state_dim: Dimensionalität des Zustandsraums
        action_dim: Dimensionalität des Aktionsraums
        dynamics_func: Funktion (state, action) -> next_state
        reward_func: Funktion (state, action) -> reward
        gamma: Diskontierungsfaktor
        """
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.dynamics_func = dynamics_func
        self.reward_func = reward_func
        self.gamma = float(gamma)
        self.state = self.reset()

    def reset(self):
        """
        Setzt das MDP auf einen zufälligen Startzustand zurück.
        """
        self.state = self.sample_state()
        return self.state

    def step(self, action):
        """
        Führt die Aktion aus, gibt (nächster Zustand, Reward) zurück.
        """
        next_state = self.dynamics_func(self.state, action)
        reward = self.reward_func(self.state, action)
        self.state = next_state
        return next_state, reward

    def sample_state(self):
        """
        Gibt einen zufälligen Zustand zurück (hier: uniform im Intervall [-1, 1]).
        """
        return np.random.uniform(-1, 1, size=self.state_dim)

    def sample_action(self):
        """
        Gibt eine zufällige Aktion zurück (hier: uniform im Intervall [-1, 1]).
        """
        return np.random.uniform(-1, 1, size=self.action_dim)

    def to_dict(self):
        """
        Gibt eine serialisierbare Repräsentation zurück.
        Die Funktionsobjekte werden nicht serialisiert!
        """
        return {
            "type": "continuous",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "has_dynamics_func": self.dynamics_func is not None,
            "has_reward_func": self.reward_func is not None
        }

    @classmethod
    def from_dict(cls, data):
        """
        Kann nicht sinnvoll die Funktionen rekonstruieren.
        """
        raise NotImplementedError("Funktionen können nicht automatisch aus dict rekonstruiert werden. Bitte init mit eigenen Funktionen verwenden.")
