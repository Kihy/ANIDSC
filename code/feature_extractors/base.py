from abc import ABC, abstractmethod

class BaseTrafficFeatureExtractor(ABC):
    
    @abstractmethod
    def update(self, traffic_vector):
        pass
    
    @abstractmethod
    def peek(self, traffic_vectors):
        pass
    
    @abstractmethod
    def get_traffic_vector(self, packet):
        pass
    
    @abstractmethod
    def setup(self, **kwargs):
        pass


    @abstractmethod
    def get_headers(self):
        pass

    @abstractmethod
    def teardown(self):
        pass
    
    @abstractmethod
    def extract_features(self):
        pass
