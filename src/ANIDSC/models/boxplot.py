import numpy as np
from ..base_files import BaseOnlineODModel
from pytdigest import TDigest


class BoxPlot(BaseOnlineODModel):
    def __init__(self, **kwargs):
        """a simple boxplot that detects based on frequency

        Args:
            device (str, optional): device for this model. Defaults to "cuda".
            node_encoder (Dict[str,Any], optional): the node encoder to encode features. Defaults to None.
        """        
        BaseOnlineODModel.__init__(self, **kwargs)
    
    def get_total_params(self):
        return 0
    
    def init_model(self, context):
        self.model=TDigest()
        
    def train_step(self, X, preprocess=False):
        self.model.update(X)
        self.num_trained+=1
        return 0.
    
    def predict_step(self, X, preprocess=False):
        if self.num_trained==0:
            return np.zeros(X.shape[0]), float("inf")
            
        quantiles=self.model.inverse_cdf([0.25,0.75])
        IQR=quantiles[1]-quantiles[0]
        upper_bound=quantiles[1]+1.5*IQR
        
        mid=(quantiles[1]+quantiles[0])/2 
        self.num_evaluated+=1
        return np.abs(X-mid), upper_bound-mid    
    
    def forward(self, x, inference=False):
        pass
    
    def __getstate__(self):
        state_dict=self.__dict__.copy()
        state_dict['model']=state_dict['model'].get_centroids()
        return state_dict
        
    def __setstate__(self, state):
        state['model']=TDigest.of_centroids(state['model'])
        
        self.__dict__.update(state)
        