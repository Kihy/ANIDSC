

from pathlib import Path
import pickle


class PickleSaveMixin:
    def save(self):
        """Save the object to a file using pickle.
        """ 
        super().save()
        save_path = Path(
            self.get_save_path()
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(save_path), 'wb') as file:
            pickle.dump(self, file)
        
        
                
    
    @classmethod
    def load(cls, path): # dataset_name:str, fe_name:str, file_name:str, name:str, suffix:str=''
        """Load an object from a file using pickle

        Args:
            folder (str): folder of the object
            dataset_name (str): datasetname associated
            fe_name (str): feature extractor name
            file_name (str): file name
            name (str): name of component
            suffix (str, optional): any suffix. Defaults to ''.
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

    
    