
from typing import Tuple

from ..graph_rep.base import GraphRepresentation

from ..save_mixin.pickle import PickleSaveMixin

from ..component.pipeline_component import PipelineComponent

from torch_geometric.utils import from_networkx
from networkx.readwrite import json_graph

class Plain(GraphRepresentation):
    
    def transform(self, X):
        """ Does nothing
        """
        return X 



class Filter(GraphRepresentation):
    

    def transform(self, X):
        """remove zeros
        """
        # remove nodes with all 0 values
        to_remove = [
            n for n, attrs in X.nodes(data=True)
            if attrs.get("count", 0) == 0 and attrs.get("size", 0) == 0
        ]
        
        X.remove_nodes_from(to_remove)
        
        if len(X.nodes)==0:
            return None
        else:
            return None 