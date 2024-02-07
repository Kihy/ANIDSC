from minisom import MiniSom 
from models.base_model import BaseModel
import numpy as np 

class SOM(BaseModel):
    def __init__(self, preprocessors=[], **kwargs):
        self.model=MiniSom(**kwargs)
        self.preprocessors = preprocessors
        self.model_name="SOM"
    
    def train(self, X):
        self.model.train(self.preprocess(X), 1)
    
    def predict_scores(self, X):
        X=self.preprocess(X)
        return np.linalg.norm(self.model.quantization(X) - X, axis=1)

    def predict_labels(self, X):
        if self.threshold is None:
            raise ValueError(
                "predict_labels only works after threshold is calculated. Call calc_threshold() first"
            )
        label = self.predict_scores(X) > self.threshold
        return label
    
