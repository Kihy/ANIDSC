
import numpy as np
from utils import to_numpy
from rrcf import RCTree
from models.base_model import *



class RRCF(BaseOnlineODModel, DictSaveMixin):
    def __init__(
        self,
        num_trees=10,
        tree_size=1024,
        preprocessors=[],
        **kwargs
    ):
        BaseOnlineODModel.__init__(self,preprocessors=preprocessors,**kwargs)
        self.num_trees=num_trees
        self.tree_size=tree_size
        self.model_name="RRCF"
        self.forest = []
        for _ in range(num_trees):
            tree = RCTree()
            self.forest.append(tree)
            
        self.index=0
        # self.preprocessors.append(to_numpy)
        
    def to_dict(self):
        ret={"index":self.index,
             "score_hist":self.score_hist, "data_max":self.data_max, "data_min":self.data_min,
             "forest":[]}
        for i in self.forest:
            ret["forest"].append(i.to_dict())
        return ret
            
    def load_dict(self, d):
        for key, value in d.items():
            if key=="forest":
                self.forest=[]
                for i in value: 
                    tree = RCTree()
                    tree.load_dict(i)
                    self.forest.append(tree)
            else:            
                setattr(self, key, value)
            
        
    def process(self, X):
        threshold=self.get_threshold()
        X=self.preprocess(X)

        scores=[]
        for point in X:
            avg_codisp=0
            for tree in self.forest:
                # If tree is above permitted size...
                if len(tree.leaves) > self.tree_size:
                    # Drop the oldest point (FIFO)
                    tree.forget_point(self.index - self.tree_size)
                # Insert the new point into the tree
                tree.insert_point(point, index=self.index)
                # Compute codisp on the new point...
                new_codisp = tree.codisp(self.index)
                # And take the average over all trees
                
                avg_codisp += new_codisp / self.num_trees
            scores.append(avg_codisp)
            self.score_hist.append(avg_codisp)
            self.index+=1
        return np.array(scores), threshold