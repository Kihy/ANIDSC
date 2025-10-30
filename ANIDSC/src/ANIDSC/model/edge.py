from typing import Tuple
from ..model.model import BaseOnlineODModel
from ..model.torch_model.base_torch_model import BaseTorchModel
from ..component.pipeline_component import PipelineComponent
from ..save_mixin.torch import TorchSaveMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch

from torch_geometric.nn.models import GAT, GAE, MLP, GCN


class EdgeGNN(BaseTorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessors = [] # dont need it
        
        
        
    def sample_rows(self, z):
        """
        z: [num_rows, d] tensor
        N: number of rows to sample
        returns: [N, d] sampled rows and their indices
        """
        total = z.size(0)
        indices = torch.randperm(total)[:self.max_z]
        return z[indices]
    
    def eps_all_pairs_loss(self,z):
        """
        z: [N, d] node embeddings
        eps: float — pulling threshold
        margin: float — desired minimum distance for repulsion
        """
        if z.shape[0]>self.max_z:
            z=self.sample_rows(z)
        
        # Compute pairwise L2 distance matrix [N, N]
        dist = torch.cdist(z, z, p=2)  # shape [N, N]

        # Mask to exclude self-comparisons
        N = z.size(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

        # Apply mask
        dist = dist[mask].view(N, N - 1)

        # Pull loss: pairs with dist < eps
        pull_mask = dist < self.eps
        
        loss_mat=torch.where(pull_mask, dist, dist ** 2)



        return loss_mat.mean(1)
    
    def init_model(self):
        
        self.hidden_channels=10
        self.node_dims=10
        self.eps=10
        self.margin=1
        self.max_z=50
        
        
        self.node_embed = GAT(
            in_channels=self.input_dims,
            hidden_channels=self.hidden_channels,
            out_channels=self.node_dims,
            num_layers=2,
            norm=None,
        ).to(self.device)
        
        
        self.optimizer = torch.optim.Adam(params=self.parameters())
        
        self.node_latent={}

    def forward(self,  X, inference=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """the forward pass of the model

        Args:
            X (_type_): input data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of output and loss
        """

        labels=X.label
        node_latent=[]
        
        for i in labels:
            if i not in self.node_latent.keys():
                self.node_latent[i]=self.to_device(torch.normal(0, 1, size=(1,self.node_dims)))
            node_latent.append(self.node_latent[i])
            

        
        node_embeddings = self.node_embed(
            x=self.to_device(torch.vstack(node_latent)), edge_index=X.edge_index, edge_attr=X.mean
        )
        
        
            
        loss=self.eps_all_pairs_loss(node_embeddings)
        
        for i,label in enumerate(labels):
            self.node_latent[label]=node_embeddings[None, i].detach()
        
        return loss, loss

    

        