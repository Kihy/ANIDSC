
import json
from pathlib import Path
import pickle

import numpy as np
import torch


class PickleSaveMixin:
    def save_pickle(self, folder, suffix=""):
        """Save the object to a file using pickle."""
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
    def load_pickle(cls, folder, dataset_name, fe_name, file_name, name, suffix=''):
        """Load an object from a file using pickle."""
        
        file_path = Path(
        f"{dataset_name}/{fe_name}/{folder}/{file_name}/{name}{f'-{suffix}' if suffix !='' else ''}.pkl"
    )
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
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
    def save(self, suffix=""):
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

    def load(self, suffix=""):
        context=self.get_context()
        checkpoint = torch.load(f"{context['dataset_name']}/{context['fe_name']}/models/{context['file_name']}/{self.name}{f'-{suffix}' if suffix !='' else ''}.pth"
        )

        self.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"][0])

    def state_dict(self):
        state = super().state_dict()
        for i in self.custom_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):
        for i in self.custom_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
        super().load_state_dict(state_dict)