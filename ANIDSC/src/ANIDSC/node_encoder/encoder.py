from abc import ABC, abstractmethod

from ..model.torch_model.base_torch_model import BaseTorchModel

from ..save_mixin.pickle import PickleSaveMixin
from torch_geometric.data import Data

from ..component.pipeline_component import PipelineComponent
from ..save_mixin.torch import TorchSaveMixin
import sys 
import torch
from ..model.gnnids import SWLoss
from torch_geometric.nn.models import GAT, GAE, MLP, GCN
from ..converters import auto_cast_method

class BaseNodeEmbedder(PickleSaveMixin, PipelineComponent):
    def __init__(self, embedder_name, embedder_params=None, **kwargs):
        super().__init__(**kwargs)
        self.embedder_name=embedder_name 
        self.embedder_params=embedder_params or {}
        self.embedder=None 
        
    
    def setup(self):
        if self.embedder is None:
            
            self.embedder_cls=getattr(sys.modules[__name__], self.embedder_name)
                    
            ndim=self.request_attr("output_dim")

            self.embedder=self.embedder_cls(input_dims=ndim, **self.embedder_params)

    @auto_cast_method
    def process(self, data: Data):
        node_embeddings, _ = self.embedder.predict_step(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr
        )
        
        # assign inf to all node embeddings 
        # node_embeddings[data.malicious]=torch.inf
        
        self.embedder.train_step(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        
        return node_embeddings

    def teardown(self):
        pass
    
    @property
    def output_dim(self):
        return self.embedder.output_dim
    
    def __str__(self):
        return f"{self.embedder_name}"
    
class GNNEmbedder(BaseTorchModel):
    
    def __init__(self, hidden_channels, out_channels, num_layers, *args, **kwargs):
        
        self.hidden_channels=hidden_channels
        self.out_channels=out_channels
        self.num_layers=num_layers
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def create_node_embed(self):
        pass 
    
    def init_model(self):

        
        self.linear = torch.nn.Linear(self.out_channels, self.out_channels).to(
            self.device
        )
        self.sw_loss = SWLoss(50, "uniform").to(self.device)
        
        self.create_node_embed()
        
        self.optimizer = torch.optim.Adam(params=self.parameters())
    
    @property
    def output_dim(self):
        return self.out_channels
        
    def forward(self, x, edge_index, edge_attr):
        node_embeddings = self.node_embed(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        return self.linear(node_embeddings), self.sw_loss(node_embeddings)
    
    def predict_step(self, x, edge_index, edge_attr):
        
        with torch.no_grad():
            embedding, loss = self.forward(x, edge_index, edge_attr)
        
        return embedding, loss.detach().cpu().numpy()
    
    def train_step(self, x, edge_index, edge_attr):
        self.optimizer.zero_grad()
        
        _, loss = self.forward(x, edge_index, edge_attr)
        
        loss = loss.mean()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss.detach().cpu().item()
    

class GCNEmbedder(GNNEmbedder):
    def create_node_embed(self):
        self.node_embed = GCN(
                    in_channels=self.input_dims,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=self.num_layers,
                    norm=None,
                ).to("cuda")
        
class GATEmbedder(GNNEmbedder):
    def create_node_embed(self):
        self.node_embed = GAT(
                    in_channels=self.input_dims,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=self.num_layers,
                    norm=None,
                ).to("cuda")
        
class MLPEmbedder(GNNEmbedder):
    def create_node_embed(self):
        self.node_embed = MLP(
                    in_channels=self.input_dims,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=self.num_layers,
                    norm=None,
                ).to("cuda")
        
    def forward(self, x, edge_index, edge_attr):
        node_embeddings = self.node_embed(
            x=x
        )
        return self.linear(node_embeddings), self.sw_loss(node_embeddings)
    

class PassThroughEmbedder:
    def __init__(self, input_dims):
        self.input_dims=input_dims 
    
    @property
    def output_dims(self):
        return self.input_dims
    
    def predict_step(self, x, edge_index, edge_attr):
        return x, None

    def train_step(self, x, edge_index, edge_attr):
        return None 
    
    def __eq__(self, value):
        return True

