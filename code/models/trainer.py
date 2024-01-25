from models.sklearn_models import *
from datasets.custom_dataset import * 
import pickle

class BasePipeline(ABC):

    def setup_dataset(self):
        self.dataset_id=self.files["benign"]
        self.benign_dataset=load_dataset(self.files["benign"], train_val_test=self.train_val_test, batch_size=self.batch_size)        
        self.malicious_datasets=[load_dataset(path, train_val_test=False,batch_size=self.batch_size) for path in self.files["malicious"]]
        
    def start(self, **kwargs):
        for k, v in kwargs.items():
            assert(k in self.allowed)
            setattr(self, k, v)
        
        self.setup_dataset()
        results={}
        for step in self.steps:
            func = getattr(self, step)
            results[step]=func()
        return results
    
    def __rrshift__(self, other):    
        return self.start(**other)
    
    def save(self):
        self.model.save(self.dataset_id)

    def eval(self):
        scores=[]
        for feature, label in self.benign_dataset["test"]:
            scores.append(self.model.predict_scores(feature))
            
        model_output={f"{self.dataset_id}_test":np.hstack(scores),
                      "threshold":self.model.threshold}
        
        for name, dataset in zip(self.files["malicious"], self.malicious_datasets):
            scores=[]
            for feature, label in dataset:
                scores.append(self.model.predict_scores(feature))
            model_output[name]=np.hstack(scores)
        
        results={}
        for metric in self.metrics:
            results[metric.__name__]=metric(model_output)
            
        return results
    
    def train(self):
        for feature, label in self.benign_dataset["train"]:
            self.model.train(feature)


class OutlierDetectionPipeline(BasePipeline):

    def __init__(self, batch_size=1024, 
                 train_val_test=[0.8,0.1,0.1], 
                 steps=["train","calc_threshold","eval","save"],**kwargs):
        
        self.allowed= ("metrics", "model", "files")

        for k, v in kwargs.items():
            assert( k in self.allowed )
            setattr(self, k, v)
        self.steps=steps
        self.batch_size=batch_size
        self.train_val_test=train_val_test
        
    def calc_threshold(self, func=lambda x: np.percentile(x, 99.9)):
        scores=[]
        for feature, label in self.benign_dataset["val"]:
            scores.append(self.model.predict_scores(feature))
        threshold=func(np.hstack(scores))
        self.model.threshold=threshold
        return threshold


    
    
        