from abc import ABC, abstractmethod

from ..model.torch_model.base_torch_model import BaseTorchModel

from ..save_mixin.pickle import PickleSaveMixin

from ..component.pipeline_component import PipelineComponent
from ..save_mixin.torch import TorchSaveMixin
import sys 
import torch
from ..model.gnnids import SWLoss
from torch_geometric.nn.models import GAT, GAE, MLP, GCN


class BaseNodeEmbedder(PickleSaveMixin, PipelineComponent):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name=model_name 
        self.model=None 
    
    def setup(self):
        if self.model is None:
            
            self.model_cls=getattr(sys.modules[__name__], self.model_name)
                    
            ndim=self.request_attr("output_dim")

            self.model=self.model_cls(ndim)

    def process(self, data):
        node_embeddings, _ = self.model.predict_step(
            x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr
        )
        
        self.model.train_step(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        
        return node_embeddings

    def teardown(self):
        pass
    
    @property
    def output_dim(self):
        return self.model.output_dim
    
    def __str__(self):
        return f"BaseNodeEmbedder({self.model_name})"
    
class GNNEmbedder(BaseTorchModel):
        
    @abstractmethod
    def create_node_embed(self):
        pass 
    

    
    def init_model(self):
        self.hidden_channels=5
        self.out_channels=10
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
        embedding, loss = self.forward(x, edge_index, edge_attr)
        

        return embedding, loss.detach().cpu().numpy()
    
    def train_step(self, x, edge_index, edge_attr):
        self.optimizer.zero_grad()
        
        _, loss = self.forward(x, edge_index, edge_attr)
        
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        
        return loss.detach().cpu().item()
    

class GCNEmbedder(GNNEmbedder):
    def create_node_embed(self):
        self.node_embed = GCN(
                    in_channels=self.input_dims,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=2,
                    norm=None,
                ).to("cuda")
        
class GATEmbedder(GNNEmbedder):
    def create_node_embed(self):
        self.node_embed = GAT(
                    in_channels=self.input_dims,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=2,
                    norm=None,
                ).to("cuda")
        
class MLPEmbedder(GNNEmbedder):
    def create_node_embed(self):
        self.node_embed = MLP(
                    in_channels=self.input_dims,
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_layers=2,
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


