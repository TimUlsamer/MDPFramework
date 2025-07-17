# mdp_framework/core/base.py

from abc import ABC, abstractmethod

class BaseMDP(ABC):
    """
    Abstrakte Basisklasse für alle MDP-Typen.
    """

    @abstractmethod
    def reset(self):
        """
        Setzt das MDP auf einen Startzustand zurück und gibt diesen Zustand zurück.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Führt die gegebene Aktion aus, liefert (nächster Zustand, Reward).
        """
        pass

    @abstractmethod
    def sample_state(self):
        """
        Gibt einen zufälligen Zustand zurück.
        """
        pass

    @abstractmethod
    def sample_action(self):
        """
        Gibt eine zufällige Aktion zurück.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Gibt eine serialisierbare Repräsentation zurück (für JSON, YAML, etc.).
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data):
        """
        Erzeugt eine MDP-Instanz aus serialisierten Daten.
        """
        pass
