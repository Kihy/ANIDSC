from abc import abstractmethod
import importlib
import torch
from torch_geometric.data import Data
from ...converters.decorator import auto_cast_method
import numpy as np 
import torch.nn.utils as nn_utils


class BaseTorchModel(torch.nn.Module):
    def __init__(self, input_dims, device="cuda", **kwargs):
        super().__init__()
        self.preprocessors = [self.to_float, self.to_device]
        self.device = device
        self.optimizer = None
        self.input_dims=input_dims

        self.init_model()

    @abstractmethod    
    def init_model(self):
        pass

    def preprocess(self, X):
        """preprocesses the input with preprocessor

        Args:
            X (_type_): input data

        Returns:
            _type_: preprocessed X
        """
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = p(X)
        return X

    def to_device(self, X: torch.Tensor) -> torch.Tensor:
        return X.to(self.device)

    def to_float(self, X: torch.Tensor)-> torch.Tensor:
        return X.float()

    def to_numpy(self, X: torch.Tensor)-> np.ndarray:
        return X.detach().cpu().numpy()

    @abstractmethod
    def forward(self, X, inference=False):
        pass

    @auto_cast_method
    def predict_step(self, X:torch.Tensor)-> np.ndarray:
                
        X = self.preprocess(X)

        with torch.no_grad():
            _, loss = self.forward(X, inference=True)
        
        if loss is None:
            return None

        return self.to_numpy(loss)

    @auto_cast_method
    def train_step(self, X:torch.Tensor)-> np.ndarray:
        X = self.preprocess(X)
        
        # Kitsune creates optimizer later
        if self.optimizer is None:
            return 
        
        self.optimizer.zero_grad()
        
        _, loss = self.forward(X, inference=False)

        if loss is None:
            return None
        
        mean_loss = loss.mean()

        mean_loss.backward()
        

        self.optimizer.step()
        
        return self.to_numpy(loss)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        return self.__class__.__name__
    

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the model's full state_dict (includes all parameters/buffers)
        state["model_state_dict"] = self.state_dict()
        return state

    def __setstate__(self, state):
        # Extract the model's state_dict and restore it
        model_state_dict = state.pop("model_state_dict")
        self.__dict__.update(state)
        self.load_state_dict(model_state_dict)
            
