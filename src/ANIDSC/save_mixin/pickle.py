

from pathlib import Path
import pickle


class PickleSaveMixin:
    def save(self):
        """Save the object to a file using pickle.
        """        
        save_path = Path(
            f"{self.context['dataset_name']}/{self.context['fe_name']}/{self.component_type}/{self.context['file_name']}/{self.__str__()}{f'-'.join(self.suffix)}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # no need parent reference
        self.parent=None

        with open(str(save_path), 'wb') as file:
            pickle.dump(self, file)
        print(f"{self.component_type} component saved to {save_path}")
    
    @classmethod
    def load(cls, dataset_name:str, fe_name:str, file_name:str, name:str, suffix:str=''):
        """Load an object from a file using pickle

        Args:
            folder (str): folder of the object
            dataset_name (str): datasetname associated
            fe_name (str): feature extractor name
            file_name (str): file name
            name (str): name of component
            suffix (str, optional): any suffix. Defaults to ''.
        """        
        file_path = Path(
        f"{dataset_name}/{fe_name}/{cls.component_type}/{file_name}/{name}{f'-{suffix}' if suffix !='' else ''}.pkl"
    )   
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        
        obj.loaded_from_file=True
        print(f"Object loaded from {file_path}")
        return obj

    
    