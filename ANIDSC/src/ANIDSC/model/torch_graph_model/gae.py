from typing import Tuple
from ANIDSC.converters.decorator import auto_cast_method
import numpy as np
from .base_model import BaseGraphModel
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    GATConv,
    GINConv,
    global_add_pool,
)
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch.nn import Sequential, ReLU, ModuleList, Linear

from torch_geometric.data import Data


class StructuralGAE(BaseGraphModel):
    def __init__(self, hidden_dim, latent_dim, *args, **kwargs):
        """a base autoencoder

        Args:
            device (str, optional): device for this model. Defaults to "cuda".
            node_encoder (Dict[str,Any], optional): the node encoder to encode features. Defaults to None.
        """
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        super().__init__(*args, **kwargs)

    def init_model(self):

        # Encoder: node features -> latent representation
        self.initial_lin = Linear(1, self.hidden_dim).to(self.device)

        self.conv1 = GCNConv(self.hidden_dim, self.hidden_dim).to(self.device)
        self.conv2 = GCNConv(self.hidden_dim, self.latent_dim).to(self.device)

        self.optimizer = torch.optim.Adam(params=self.parameters())

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features to latent representation

        Args:
            x: Node features [num_nodes, input_dims]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            Latent node embeddings [num_nodes, latent_dims]
        """
        x = self.initial_lin(x)
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstructed node features

        Args:
            z: Latent node embeddings [num_nodes, latent_dims]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            Reconstructed node features [num_nodes, input_dims]
        """
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def forward(self, data: Data, inference=False) -> Tuple[torch.Tensor, torch.Tensor]:

        x, edge_index = data.x, data.edge_index

        # encode
        z = self.encode(x, edge_index)
        graph_emb = global_mean_pool(z, batch=None)

        return z, graph_emb

    @auto_cast_method
    def train_graph(self, data: Data) -> np.ndarray:
        """Train on a single graph data object"""

        z, _ = self.forward(data)

        # decode postive edges
        pos_pred = self.decode(z, data.edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred)
        )

        # decode negative edges
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )

        neg_pred = self.decode(z, neg_edge_index)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred, torch.zeros_like(neg_pred)
        )

        # Calculate reconstruction loss per node
        loss = pos_loss + neg_loss
        loss.backward()
        self.optimizer.step()

        return self.to_numpy(loss)

    @auto_cast_method
    def predict_graph(self, data: Data) -> np.ndarray:
        z, _ = self.forward(data)
        pos_edge_index = data.edge_index
        pos_pred = self.decode(z, pos_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred), reduction="mean"
        )
        return self.to_numpy(pos_loss)


class GINGAE(StructuralGAE):
    def __init__(self, num_layers, *args, **kwargs):
        """a base autoencoder

        Args:
            device (str, optional): device for this model. Defaults to "cuda".
            node_encoder (Dict[str,Any], optional): the node encoder to encode features. Defaults to None.
        """
        self.num_layers = num_layers
        super().__init__(*args, **kwargs)

    def init_model(self):

        # Encoder: node features -> latent representation
        self.initial_lin = Linear(1, self.hidden_dim).to(self.device)

        self.convs = ModuleList().to(self.device)
        for _ in range(self.num_layers - 1):
            nn_mlp = Sequential(
                Linear(self.hidden_dim, self.hidden_dim),
                ReLU(),
                Linear(self.hidden_dim, self.hidden_dim),
            ).to(self.device)
            self.convs.append(GINConv(nn_mlp).to(self.device))

        # Final layer to latent space
        nn_mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.latent_dim),
        ).to(self.device)
        self.convs.append(GINConv(nn_mlp).to(self.device))

        self.optimizer = torch.optim.Adam(params=self.parameters())

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features to latent representation

        Args:
            x: Node features [num_nodes, input_dims]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            Latent node embeddings [num_nodes, latent_dims]
        """
        x = self.initial_lin(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

    def forward(self, data: Data, inference=False) -> Tuple[torch.Tensor, torch.Tensor]:

        x, edge_index = data.x, data.edge_index

        # encode
        z = self.encode(x, edge_index)
        graph_emb = global_add_pool(z, batch=None)

        return z, graph_emb
