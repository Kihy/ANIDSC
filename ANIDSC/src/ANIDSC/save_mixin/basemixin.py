from abc import abstractmethod

from pathlib import Path 
class BaseSaveMixin:
    
    @property
    def save_path(self):
        path=f"{self.request_attr('dataset_name')}/{self.request_attr('fe_name')}/saved_components/{self.request_attr('file_name')}/{self.component_type}/{str(self)}.{self.save_type}"
        return Path(path)
    
    @property
    @abstractmethod
    def save_type(self):
        pass
    
    @abstractmethod
    def save(self):
        pass 
    
    @classmethod
    @abstractmethod
    def load(cls, path):
        pass 