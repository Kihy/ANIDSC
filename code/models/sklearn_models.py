import importlib
import sklearn
from datasets.custom_dataset import *
from abc import ABC, abstractmethod
from pathlib import Path
import pickle

def load_sklearn_model(dataset_id, model_name):
    with open(f"../trained_models/{dataset_id}/{model_name}.pkl","rb") as f:
        model=pickle.load(f) 
    return model

class BaseSklearnModel:
    """
    compatible with all sklearn style models
    for example pyod or deepod
    """
    def preprocess(self, X):        
        if len(self.preprocessors)>0:
            for p in self.preprocessors:
                X=p(X)
        return X
    
    @abstractmethod
    def train(self, X):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def save(self, dataset_id):
        save_path=Path(f"../trained_models/{dataset_id}/{self.model_name}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path,"wb") as f:
            pickle.dump(self, f) 

class SklearnClassifier(BaseSklearnModel):
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
        
        
    def predict(self,X):
        X=self.preprocess(X)
        self.model.decision_function(X)
        
class SklearnOutlierDetector(BaseSklearnModel):
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
    
        
