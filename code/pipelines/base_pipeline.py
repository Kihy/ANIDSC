from models.sklearn_models import *
from datasets.custom_dataset import *
import pickle
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import LazyInitializationMixin
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.tensorboard import SummaryWriter

class BasePipeline(ABC, LazyInitializationMixin):
    def __init__(self, 
                 allowed=["model", "dataset_name", "files"],
                 **kwargs):
        
        self.allowed = allowed
        self.lazy_init(**kwargs)
        self.entry = self.run_pipeline
    
    @abstractmethod
    def setup_dataset(self):
        pass
    
    def run_pipeline(self):
        self.setup_dataset()
        results = {}
        for step in self.steps:
            func = getattr(self, step)
            results[step] = func()
        return results
    
    def metric_names(self):
        return [metric.__name__ for metric in self.metrics]


