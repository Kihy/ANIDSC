

from pathlib import Path
from typing import Any, Dict

import torch


class TorchSaveMixin:
    def save(self, suffix:str=""):
        """save model with torch, all torch models are saved in models folder

        Args:
            suffix (str, optional): suffix of model. Defaults to "".
        """        

        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if hasattr(self, "optimizer"):
            checkpoint["optimizer_state_dict"] = (self.optimizer.state_dict(),)
        ckpt_path = Path(
            f"{self.context['dataset_name']}/{self.context['fe_name']}/models/{self.context['file_name']}/{self.component_name}{f'-{suffix}' if suffix !='' else ''}.pth"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(ckpt_path))
        print(f"model saved at {ckpt_path}")

    def load(self, load_name, suffix:str=""):
        """loads the parameters of torch model

        Args:
            suffix (str, optional): optional suffix. Defaults to "".
        """        
        ckpt_path=f"{self.context['dataset_name']}/{self.context['fe_name']}/models/{load_name}/{self.component_name}{f'-{suffix}' if suffix !='' else ''}.pth"
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