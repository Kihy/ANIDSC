
from typing import Tuple

from ..save_mixin.pickle import PickleSaveMixin

from ..component.pipeline_component import PipelineComponent
from ..save_mixin.torch import TorchSaveMixin
import torch
from torch_geometric.utils import from_networkx
from networkx.readwrite import json_graph




class PlainGraphRepresentation(PickleSaveMixin, PipelineComponent):
    def __init__(self):
       

        super().__init__()


    @property
    def output_dim(self):
        return 2
    
    def teardown(self):
        pass

    @property 
    def networkx(self):
        return self.graph

    def setup(self):
        super().setup()

    def process(self, X):
        """converts json format to pytorch_geometric Data format.
        """

        X = json_graph.node_link_graph(X)
        self.graph=X
        
        data = from_networkx(X, group_node_attrs=["count","size"])
        data.x=data.x.float()
        
        data.label = [node for node in X.nodes]
        data.time_stamp = X.graph["time_stamp"]

        return data.to("cuda")

