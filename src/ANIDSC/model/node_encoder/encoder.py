from abc import ABC, abstractmethod

from ...component.pipeline_component import PipelineComponent
from ...save_mixin.torch import TorchSaveMixin
from ..gnnids import SWLoss
import torch

from torch_geometric.nn.models import GAT, GAE, MLP, GCN


class BaseNodeEncoder(TorchSaveMixin, PipelineComponent, torch.nn.Module):
    def __init__(
        self,
        n_features:int,
        node_latent_dim: int = 15,
        embedding_dist: str = "uniform",
        device: str = "cuda",
    ):
        torch.nn.Module.__init__(self)
        PipelineComponent.__init__(self)
        self.node_latent_dim = node_latent_dim
        self.embedding_dist = embedding_dist
        self.device = device
        self.custom_params=[]
        
        self.linear = torch.nn.Linear(self.node_latent_dim, self.node_latent_dim).to(
            self.device
        )
        self.n_features=n_features
        self.sw_loss = SWLoss(50, self.embedding_dist).to(device)

        self.save_attr.extend(["n_features","node_latent_dim", "embedding_dist", "device"])

        
    def process(self, data):
        embedding, sw_loss=self.forward(*data)
        
        return embedding

    def forward(self, x, edge_index, edge_attr=None):
        node_embeddings = self.node_embed(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        return self.linear(node_embeddings), self.sw_loss(node_embeddings)


class GATNodeEncoder(BaseNodeEncoder):
    def setup(self):
        
        self.node_embed = GAT(
            in_channels=self.n_features,
            hidden_channels=self.n_features,
            out_channels=self.node_latent_dim,
            num_layers=2,
            norm=None,
        ).to(self.device)


class LinearNodeEncoder(BaseNodeEncoder):
    def setup(self):
        
        self.node_embed = MLP(
            in_channels=self.n_features,
            hidden_channels=self.n_features,
            out_channels=self.node_latent_dim,
            num_layers=2,
            norm=None,
        ).to(self.device)

    def forward(self, x, edge_index, edge_attr=None):
        node_embeddings = self.node_embed(x=x)
        return self.linear(node_embeddings), self.sw_loss(node_embeddings)


class GCNNodeEncoder(BaseNodeEncoder):
    def setup(self):
        
        self.node_embed = GCN(
            in_channels=self.n_features,
            hidden_channels=self.n_features,
            out_channels=self.node_latent_dim,
            num_layers=2,
            norm=None,
        ).to(self.device)
