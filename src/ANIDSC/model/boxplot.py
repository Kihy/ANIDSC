from ..component.pipeline_component import PipelineComponent
from ..save_mixin.pickle import PickleSaveMixin
import numpy as np

from pytdigest import TDigest


class BoxPlot(PickleSaveMixin, PipelineComponent):
    def __init__(self, **kwargs):
        """a simple boxplot that detects based on frequency, the input_dim must be 1
        """
        super().__init__(component_type="model", **kwargs)
        self.model=TDigest()
        self.batch_trained = 0
        self.batch_evaluated = 0

    def setup(self):
        pass 
    
    def get_total_params(self):
        return 0
    
    def process(self, X):
        
        if np.isnan(self.model.mean):
            threshold=float('inf')
            score=np.zeros(X.shape[0])
        else:
                
            quantiles=self.model.inverse_cdf([0.25,0.75])
            IQR=quantiles[1]-quantiles[0]
            upper_bound=quantiles[1]+1.5*IQR
            
            mid=(quantiles[1]+quantiles[0])/2 
            threshold=upper_bound-mid
            score=np.abs(X-mid)
        
        self.batch_evaluated+=1
        self.batch_trained+=1
        # update model
        self.model.update(X)

        return {"threshold": threshold , "score": score, "batch_num": self.batch_evaluated}


         
    def __getstate__(self):
        state_dict=super().__getstate__()
        state_dict['model']=state_dict['model'].get_centroids()
        return state_dict
        
    def __setstate__(self, state):
        state['model']=TDigest.of_centroids(state['model'])
        
        super().__setstate__(state)
        