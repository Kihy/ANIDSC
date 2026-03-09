

from abc import abstractmethod
from pathlib import Path
import pickle
import sys

from ..save_mixin.basemixin import BaseSaveMixin


class PickleSaveMixin(BaseSaveMixin):

    @property
    def save_type(self):
        return "pkl"
    
    def save(self):
        """Save the object to a file using pickle.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(self.save_path), 'wb') as file:
            pickle.dump(self, file)
        
        return str(self.save_path)
        
                
    
    @classmethod
    def load(cls, path):
        """Load an object from a file using pickle

        Args:
            path: Path to the pickle file
        """
        file_path = Path(path)
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        
        print(f"Object loaded from {file_path}")
        return obj

    
    