
import json
from pathlib import Path
import pickle
from typing import Any, Dict

import numpy as np
import torch


class PickleSaveMixin:
    def save_pickle(self, folder:str, suffix:str=""):
        """Save the object to a file using pickle.

        Args:
            folder (str): folder for saving
            suffix (str, optional): suffix to the model. Defaults to "".
        """        
        
        context=self.get_context()
        save_path = Path(
            f"{context['dataset_name']}/{context['fe_name']}/{folder}/{context['file_name']}/{self.name}{f'-{suffix}' if suffix !='' else ''}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # no need parent reference
        self.parent=None

        with open(str(save_path), 'wb') as file:
            pickle.dump(self, file)
        print(f"{folder} component saved to {save_path}")
        
    @classmethod
    def load_pickle(cls, folder:str, dataset_name:str, fe_name:str, file_name:str, name:str, suffix:str=''):
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
        f"{dataset_name}/{fe_name}/{folder}/{file_name}/{name}{f'-{suffix}' if suffix !='' else ''}.pkl"
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

    
    

class JSONSaveMixin:
    def save_json(self, folder, suffix=""):
        context=self.get_context()
        save_path = Path(
            f"{context['dataset_name']}/{context['fe_name']}/{folder}/{context['file_name']}/{self.name}{f'-{suffix}' if suffix !='' else ''}.json"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self, f, default=custom_default)

        print(f"saved at {save_path}")
        
    @classmethod
    def load_json(cls, folder, dataset_name, fe_name, file_name, name, suffix=''):
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



class TorchSaveMixin:
    def save(self, suffix:str=""):
        """save model with torch, all torch models are saved in models folder

        Args:
            suffix (str, optional): suffix of model. Defaults to "".
        """        
        context=self.get_context()
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if hasattr(self, "optimizer"):
            checkpoint["optimizer_state_dict"] = (self.optimizer.state_dict(),)
        ckpt_path = Path(
            f"{context['dataset_name']}/{context['fe_name']}/models/{context['file_name']}/{self.name}{f'-{suffix}' if suffix !='' else ''}.pth"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(ckpt_path))
        print(f"model saved at {ckpt_path}")

    def load(self, load_name, suffix:str=""):
        """loads the parameters of torch model

        Args:
            suffix (str, optional): optional suffix. Defaults to "".
        """        
        context=self.get_context()
        ckpt_path=f"{context['dataset_name']}/{context['fe_name']}/models/{load_name}/{self.name}{f'-{suffix}' if suffix !='' else ''}.pth"
        checkpoint = torch.load(ckpt_path)

        self.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"][0])
            
        print(f'Model loaded from {ckpt_path}')

    def state_dict(self)->Dict[str, Any]:
        """state dict of model

        Returns:
            Dict[str, Any] : state dictionary
        """        
        state = super().state_dict()
        for i in self.custom_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict:Dict[str, Any]):
        """loads the state dictionary

        Args:
            state_dict (Dict[str, Any]): the state dictionary
        """        
        for i in self.custom_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
        super().load_state_dict(state_dict)