from abc import ABC, abstractmethod
from pathlib import Path
import pickle 
import torch
from collections import deque
import numpy as np

def load_pkl_model(dataset_id, model_name):
    with open(f"../../models/{dataset_id}/{model_name}.pkl","rb") as f:
        model=pickle.load(f) 
    return model

def load_torch_model(model, dataset_id, model_name):
    checkpoint = torch.load(f"../../models/{dataset_id}/{model_name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

class PickleSaveMixin:
    def save(self, dataset_name, suffix=""):
        save_path=Path(f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path,"wb") as f:
            pickle.dump(self, f) 
            
        print(f"saved at {save_path}")
        
class TorchSaveMixin:
    def save(self, dataset_name, suffix=""):
        checkpoint = {
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
        ckpt_path=Path(f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pth")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(ckpt_path))
        
    def state_dict(self):
        state = super().state_dict()
        for i in self.additional_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):
        for i in self.additional_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
        
        super().load_state_dict(state_dict)

class BaseModel(ABC):
    def __init__(self, **kwargs):
        
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def preprocess(self, X):     
        if len(self.preprocessors)>0:
            for p in self.preprocessors:
                X=p(X)
        return X
    
class BaseOnlineODModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_hist=deque(maxlen=50000)
        self.data_max=None 
        self.data_min=None

        self.additional_params=["preprocessors", "scores_hist"]
        self.preprocessors.append(self.normalize)
    
    @abstractmethod        
    def process(self, X):
        pass
    
    def get_threshold(self):
        if len(self.score_hist) == 0:
            threshold=0
        else:
            threshold=np.percentile(self.score_hist, 99.9)
        return threshold
    
    def normalize(self, X):
        maximum=torch.max(X, 0).values
        minimum=torch.min(X, 0).values
        
        if self.data_max is None:
            self.data_max = maximum
            self.data_min = minimum
        
        X = torch.nan_to_num((X - self.data_min) / (self.data_max - self.data_min))
        
        self.data_max=torch.where(maximum>self.data_max, maximum, self.data_max)
        self.data_min=torch.where(minimum<self.data_min, minimum, self.data_min)
        
        return X

class BaseODModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._threshold=None
    
    @abstractmethod
    def train_step(self, X):
        pass
    
    @abstractmethod
    def predict_scores(self, X):
        pass
    
    @abstractmethod
    def predict_labels(self, X):
        pass
    
    def on_train_begin(self):
        pass
    
    def on_train_end(self):
        pass
    
    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
        
    def predict_labels(self, X):
        if self.threshold is None:
            raise ValueError(
                "predict_labels only works after threshold is calculated. Call calc_threshold() first"
            )
        return self.predict_scores(X) > self.threshold