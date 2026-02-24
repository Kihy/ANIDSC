from abc import abstractmethod
import importlib
from typing import List
import torch
from torch_geometric.data import Data
from ...converters.decorator import auto_cast_method
import numpy as np 


class BaseGraphModel(torch.nn.Module):
    def __init__(self, input_dims, device="cuda", **kwargs):
        super().__init__()

        self.input_dims = input_dims
        self.device = device
        self.optimizer = None
        self.init_model()

    @abstractmethod    
    def init_model(self):
        """Initialize your graph neural network layers here"""
        pass

    def to_device(self, X):
        return X.to(self.device)

    @abstractmethod
    def forward(self, data: Data, inference=False):
        """
        Args:
            data: PyTorch Geometric Data object with attributes:
                - data.x: Node features [num_nodes, num_features]
                - data.edge_index: Graph connectivity [2, num_edges]
                - data.edge_attr: Edge features (optional)
                - data.batch: Batch assignment vector (for batched graphs)
        
        Returns:
            tuple: (predictions, loss)
        """
        pass
    
    @auto_cast_method
    def predict_graph(self, data: Data) -> np.ndarray:
        """Predict on a single graph data object"""
        with torch.no_grad():
            preds, loss = self.forward(data, inference=True)
            loss=loss.mean()
        return self.to_numpy(loss)


    def predict_step(self, data: List[Data]) -> np.ndarray:
        """
        Single prediction step for graph data
        
        Args:
            data: PyTorch Geometric Data or Batch object
            
        Returns:
            numpy array of predictions
        """
        
        return_preds = []
        for d in data:
            return_preds.append(self.predict_graph(d))

        return np.array(return_preds)

    @auto_cast_method
    def train_graph(self, data: Data) -> np.ndarray:
        """Train on a single graph data object"""
        self.optimizer.zero_grad()
        _, loss = self.forward(data, inference=False)
        if loss is not None:
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
        return self.to_numpy(loss)

    def train_step(self, data: List[Data]) -> np.ndarray:
        """
        Single training step for graph data
        
        Args:
            data: PyTorch Geometric Data or Batch object
            
        Returns:
            numpy array of training loss
        """
      
        return_loss = []
        for d in data:
            return_loss.append(self.train_graph(d))
        
        return np.array(return_loss)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        return tensor.detach().cpu().numpy()

    def get_total_params(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        return self.__class__.__name__
    
    def __getstate__(self):
        """Custom pickling to save model state"""
        state = self.__dict__.copy()
        # Save the model's full state_dict (includes all parameters/buffers)
        state["model_state_dict"] = self.state_dict()
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore model state"""
        # Extract the model's state_dict and restore it
        model_state_dict = state.pop("model_state_dict")
        self.__dict__.update(state)
        self.load_state_dict(model_state_dict)