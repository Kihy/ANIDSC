
from typing import Tuple

from ..component.pipeline_component import PipelineComponent
from ..save_mixin.torch import TorchSaveMixin
import torch
from torch_geometric.utils import from_networkx
from networkx.readwrite import json_graph




class PlainGraphRepresentation(TorchSaveMixin, PipelineComponent, torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        **kwargs,
    ):
        """Basic graph representation of network

        Args:
            device (str, optional): name of device. Defaults to "cuda".
            preprocessors (List[str], optional): list of preprocessors. Defaults to [].
        """
        torch.nn.Module.__init__(self)
        PipelineComponent.__init__(self, component_type="graph_rep", **kwargs)
        self.device = device
        self.preprocessors.extend(
            ["to_pytorch_geometric_data", "to_device"]
        )
        self.custom_params = ["graph"]
        self.n_features=2
        
    def preprocess(self, X):
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = getattr(self, p)(X)
                if X is None:
                    return None
        return X

       

    def to_device(self, X):
        return X.to(self.device)

    def to_pytorch_geometric_data(self, X):
        # remove unscaled nodes for model
        X = json_graph.node_link_graph(X)
        self.graph=X
        
        data = from_networkx(X, group_node_attrs=["count","size"])
        data.x=data.x.float()
        
        data.label = [node for node in X.nodes]
        data.time_stamp = X.graph["time_stamp"]

        return data

    def get_networkx(self):
        return self.graph

    def setup(self):
        super().setup()

    def process(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """updates data with x and output graph representation after update

        Args:
            x (_type_): input data features

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: tuple of node features, edge indices, edge features
        """

        return x

