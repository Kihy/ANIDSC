from abc import ABC, abstractmethod
from pathlib import Path
import pickle 

def load_pkl_model(dataset_id, model_name):
    with open(f"../trained_models/{dataset_id}/{model_name}.pkl","rb") as f:
        model=pickle.load(f) 
    return model

class BaseModel:
    def preprocess(self, X):        
        if len(self.preprocessors)>0:
            for p in self.preprocessors:
                X=p(X)
        return X
    
    @abstractmethod
    def train(self, X):
        pass
    
    @abstractmethod
    def predict_scores(self, X):
        pass
    
    @abstractmethod
    def predict_labels(self, X):
        pass
    
    def save(self, dataset_id):
        save_path=Path(f"../trained_models/{dataset_id}/{self.model_name}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path,"wb") as f:
            pickle.dump(self, f) 
