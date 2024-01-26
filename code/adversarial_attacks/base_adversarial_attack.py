from abc import ABC, abstractmethod

class BaseAdversarialAttack(ABC):
    @abstractmethod
    def craft_adversary(self, mal_pcap):
        pass
    
