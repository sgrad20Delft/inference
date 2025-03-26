from abc import ABC, abstractmethod

class UnifiedEnergyLoggerInterface(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_total_energy(self):
        pass
