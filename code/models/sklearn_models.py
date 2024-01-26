import importlib
import sklearn
from datasets.custom_dataset import *
import pickle
from models.base_model import BaseModel

class SklearnClassifier(BaseModel):
    def __init__(self, model_name, import_module, preprocessors=[], model_params={},
                 classes=[]):
        model_class = getattr(importlib.import_module(import_module), model_name)
        self.model = model_class(**model_params)  # Instantiates the model
        self.classes=classes
        self.preprocessors=preprocessors
        self.model_name=model_name
        
        if hasattr(self.model,"partial_fit"):
            self.training_call=getattr(self.model, "partial_fit")
        else:
            self.training_call=getattr(self.model, "fit")
    
    def train(self,X):
        for features, labels in X:
            features=self.preprocess(features) 
            self.training_call(features, labels, classes=self.classes)
        
        
    def predict_scores(self,X):
        X=self.preprocess(X)
        self.model.decision_function(X)
        
class SklearnOutlierDetector(BaseModel):
    """
    compatible with all sklearn style models
    for example pyod or deepod
    """
    def __init__(self, model_name, import_module, preprocessors=[], model_params={}):
        model_class = getattr(importlib.import_module(import_module), model_name)
        self.model = model_class(**model_params)  # Instantiates the model
        self.preprocessors=preprocessors
        self.model_name=model_name
        self.threshold=None
        if hasattr(self.model,"partial_fit"):
            self.training_call=getattr(self.model, "partial_fit")
        else:
            self.training_call=getattr(self.model, "fit")
    
    def train(self, X):
        X=self.preprocess(X)  
        self.training_call(X)
        
    def predict_scores(self, X):
        X=self.preprocess(X)
        scores=-self.model.decision_function(X)
        return scores
    
    def predict_labels(self, X):
        if self.threshold is None:
            raise ValueError("predict_labels only works after threshold is calculated. Call calc_threshold() first")
        label=self.predict_scores(X)>self.threshold
        return label
    
        
