from ..base_files import BaseOnlineNormalizer
from pytdigest import TDigest
import numpy as np
from ..models.base_model import JSONSaveMixin


class LivePercentile(BaseOnlineNormalizer, JSONSaveMixin):
    def __init__(self, p=[0.25, 0.5, 0.75], load_state=False, **kwargs):
        """
        Constructs a LiveStream object
        """
        BaseOnlineNormalizer.__init__(self, **kwargs)
        self.p = p
        self.patience = 0
        self.load_state=load_state
        self.dims=[]
        
    def setup(self):
        super().setup()
        
        if self.load_state:
            self.load(folder="scalers")
        else:
            self.dims=[TDigest() for _ in range(self.ndim-self.skip)]
    
    def teardown(self):
        context=self.get_context()
        self.save(folder="scalers", suffix=context.get('protocol',''))

    def add(self, X):
        """Adds another datum"""
        
        for i, n in enumerate(X.T):
            self.dims[i].update(n)

        self.patience += 1

    def process(self, X):
        percentiles = self.quantiles()

        no_scale=X[:,:self.skip]
        scale=X[:,self.skip:]
        
        if percentiles is None:
            percentiles = np.quantile(scale, self.p, axis=0)
            
            
        scaled_features = (scale - percentiles[1]) / (percentiles[2] - percentiles[0])
        scaled_features = np.nan_to_num(
            scaled_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        self.add(scale)

        return np.hstack((no_scale, scale))

    def reset(self):
        self.dims = [TDigest() for _ in range(self.ndim-self.skip)]
        self.patience = 0

    def quantiles(self):
        """Returns a list of tuples of the quantile and its location"""

        if self.ndim == 0 or self.patience < self.warmup:
            return None
        percentiles = np.zeros((len(self.p), self.ndim-self.skip))

        for d in range(self.ndim-self.skip):
            percentiles[:, d] = self.dims[d].inverse_cdf(self.p)

        return percentiles

    def to_centroids(self):
        return [i.get_centroids() for i in self.dims]

    def of_centroids(self, dim_list):
        return [TDigest.of_centroids(np.array(i)) for i in dim_list]

    def to_dict(self):
        return {
            'p': self.p,
            'patience': self.patience,
            'load_state': self.load_state,
            'dims': self.to_centroids()
        }
        


    @classmethod
    def from_dict(cls, dict_obj):
        obj = cls()  # Call the constructor with the dictionary values
        for key, value in dict_obj.items():
            setattr(obj, key, value)
        
        obj.dims=obj.of_centroids(dict_obj['dims'])
        return obj
    