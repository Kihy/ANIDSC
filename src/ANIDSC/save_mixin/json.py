import json
from pathlib import Path

import numpy as np


class JSONSaveMixin:
    def save(self):
        super().save()
        save_path = Path(
            f"{self.context['dataset_name']}/{self.context['fe_name']}/{self.component_type}/{self.context['file_name']}/{str(self)}{f'-{self.suffix}' if self.suffix !='' else ''}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self, f, default=custom_default)

        print(f"saved at {save_path}")
        
    @classmethod
    def load(cls, folder, dataset_name, fe_name, file_name, name, suffix=''):
        """Load an object from a file using pickle."""
        
        file_path = Path(
        f"{dataset_name}/{fe_name}/{folder}/{file_name}/{name}{f'-{suffix}' if suffix !='' else ''}.json"
    )
        with open(file_path, 'rb') as file:
            obj = json.load(file)
            
        obj=cls().from_dict(obj)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        print(f"Object loaded from {file_path}")
        return obj
    
def custom_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    # elif isinstance(obj, frozenset):
    #     return list(obj)
    else:
        return obj.to_dict()