from abc import ABC, abstractmethod


class BaseAdversarialAttack(ABC):
    @abstractmethod
    def craft_adversary(self):
        pass

    @abstractmethod
    def attack_setup(self):
        pass

    @abstractmethod
    def attack_teardown(self):
        pass