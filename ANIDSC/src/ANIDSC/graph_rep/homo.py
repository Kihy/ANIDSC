

from typing import List, Tuple
from ANIDSC.component.pipeline_component import PipelineComponent
from ANIDSC.save_mixin.torch import TorchSaveMixin
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_isolated_nodes
from ..utils.helper import uniqueXT
import networkx as nx


def find_edge_indices(edges, srcID, dstID):
    search_edges = torch.vstack([srcID, dstID])
    column_indices = []
    for col in search_edges.t():
        # Check if column col of B exists in A
        mask = torch.all(edges == col.unsqueeze(1).to(edges.device), dim=0)
        # Find the index where column col of B exists in A
        index = torch.nonzero(mask, as_tuple=False)
        if index.size(0) > 0:
            # If the column exists, append its index

            column_indices.append(index.item())
        else:
            # If the column does not exist, append None
            column_indices.append(-1)
    return torch.tensor(column_indices).long()


def edge_exists(edges, search_edges):
    label = []
    for col in search_edges.t():
        # Check if column col of B exists in A
        mask = torch.all(edges == col.unsqueeze(1).to(edges.device), dim=0)
        # Find the index where column col of B exists in A
        label.append(mask.any())
    return torch.tensor(label)


def complete_directed_graph_edges(n):
    # Create an array of node indices
    nodes = torch.arange(n)

    # Generate all possible pairs of nodes
    pairs = torch.cartesian_prod(nodes, nodes)

    # Filter out self-loops
    edges = pairs[pairs[:, 0] != pairs[:, 1]]

    return edges.T


class HomoGraphRepresentation(TorchSaveMixin, PipelineComponent, torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        **kwargs,
    ):
        """Homogeneous graph representation of network

        Args:
            device (str, optional): name of device. Defaults to "cuda".
            preprocessors (List[str], optional): list of preprocessors. Defaults to [].
        """
        torch.nn.Module.__init__(self)
        PipelineComponent.__init__(self, component_type="graph_rep",**kwargs)
        self.device = device
        self.preprocessors.extend(["to_float_tensor","to_device"])
        self.n_features = 15
        self.l_features = 15

        self.custom_params = ["G"]

    def preprocess(self, X):
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = getattr(self, p)(X)
        return X

    def to_device(self, X):
        return X.to(self.device)

    def to_float_tensor(self, X):
        return torch.from_numpy(X).float()


    def setup(self):
        super().setup()

        
        # initialize graph
        self.G = Data()

        # initialize node features
        self.G.x = torch.empty((0, self.n_features)).to(self.device)
        self.G.idx = torch.empty(0).long().to(self.device)
        self.G.updated = torch.empty(0).to(self.device)

        # initialize edge features
        self.G.edge_index = torch.empty((2, 0)).long().to(self.device)
        self.G.edge_attr = torch.empty((0, self.l_features)).to(self.device)

    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        """updates source and destination nodes

        Args:
            srcID (int): id of source
            src_feature (array): src feature
            dstID (int): destionation id
            dst_feature (array): destination features
        """
        self.G.updated = torch.zeros_like(self.G.updated).to(self.device)

        unique_nodes = torch.unique(torch.cat([srcID, dstID]))

        # find set difference unique_nodes-self.G.node_idx
        uniques, counts = torch.cat((unique_nodes, self.G.idx, self.G.idx)).unique(
            return_counts=True
        )
        difference = uniques[counts == 1]

        expand_size = len(difference)

        if expand_size > 0:
            self.G.x = torch.vstack(
                [
                    self.G.x,
                    torch.zeros((expand_size, src_feature.shape[1])).to(self.device),
                ]
            )
            self.G.idx = torch.hstack([self.G.idx, difference.to(self.device)]).long()
            self.G.updated = torch.hstack(
                [self.G.updated, torch.ones(expand_size).to(self.device)]
            )

        # update
        for i in range(len(srcID)):
            self.G.x[self.G.idx == srcID[i]] = src_feature[i]
            self.G.x[self.G.idx == dstID[i]] = dst_feature[i]

        self.G.updated = torch.isin(self.G.idx, unique_nodes)

    def get_networkx(self):
        G = to_networkx(
            self.G, node_attrs=["x", "idx", "updated"], edge_attrs=["edge_attr"]
        )
        
        # get mac_to_idx_map
        mac_to_idx_map=self.request_attr("data_source","fe_attrs")["mac_to_idx_map"]
        idx_to_mac_map={v:k for k,v in mac_to_idx_map.items()}
        
        return nx.relabel_nodes(G, idx_to_mac_map)

    def update_edges(self, srcID, dstID, edge_feature):
        """updates edges between source and destination

        Args:
            srcID (int): source ID
            dstID (int): destination ID
            edge_feature (array): edge feature
        """
        # convert ID to index
        src_idx = torch.nonzero(self.G.idx == srcID[:, None], as_tuple=False)[:, 1]
        dst_idx = torch.nonzero(self.G.idx == dstID[:, None], as_tuple=False)[:, 1]

        edge_indices = find_edge_indices(self.G.edge_index, src_idx, dst_idx)
        existing_edge_idx = edge_indices != -1

        # update found edges if there are any
        if existing_edge_idx.any():
            self.G.edge_attr[edge_indices[existing_edge_idx]] = edge_feature[
                existing_edge_idx
            ]

        num_new_edges = torch.count_nonzero(~existing_edge_idx)
        if num_new_edges > 0:
            new_edges = torch.vstack(
                [src_idx[~existing_edge_idx], dst_idx[~existing_edge_idx]]
            )
            self.G.edge_index = torch.hstack([self.G.edge_index, new_edges]).long()
            self.G.edge_attr = torch.vstack(
                [self.G.edge_attr, edge_feature[~existing_edge_idx]]
            )

    def split_data(self, x):
        """splits input data into different parts

        Args:
            x (_type_): input data

        Returns:
            _type_: graph IDs and features
        """
        
        srcIDs = x[:, 1].long()
        dstIDs = x[:, 2].long()


        src_features = x[:, 4:19].float()
        dst_features = x[:, 19:34].float()
        edge_features = x[:, 34 : 34 + self.l_features].float()

        return srcIDs, dstIDs, src_features, dst_features, edge_features

    def sample_edges(self, p):
        n = self.G.x.size(0)
        probabilities = torch.rand(n, n)

        # Get indices where probabilities are less than p and apply the mask
        indices = torch.argwhere(probabilities < p).T

        return indices, edge_exists(self.edge_idx, indices)

    def process(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """updates data with x and output graph representation after update

        Args:
            x (_type_): input data features

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: tuple of node features, edge indices, edge features
        """

        diff = x[1:, 0] - x[:-1, 0]

        split_indices = torch.nonzero(diff, as_tuple=True)[0] + 1
        split_indices = split_indices.tolist()
        split_indices = [0] + split_indices + [x.size(0)]
        split_indices = np.array(split_indices[1:]) - np.array(split_indices[:-1])
        updated = False

        for data in torch.split(x, split_indices.tolist(), dim=0):
            # update chunk
            if data[0, 0] == 1:
                srcIDs, dstIDs, src_features, dst_features, edge_features = (
                    self.split_data(data)
                )

                # find index of last update
                _, indices = uniqueXT(
                    data[:, 1:3], return_index=True, occur_last=True, dim=0
                )
                self.subset = self.update_nodes(
                    srcIDs[indices],
                    src_features[indices],
                    dstIDs[indices],
                    dst_features[indices],
                )
                self.update_edges(
                    srcIDs[indices], dstIDs[indices], edge_features[indices]
                )
                updated = True

            # delete links
            else:
                srcIDs, dstIDs, _, _, _ = self.split_data(data)

                # convert to idx
                src_idx = torch.nonzero(self.G.idx == srcIDs[:, None], as_tuple=False)[
                    :, 1
                ]
                dst_idx = torch.nonzero(self.G.idx == dstIDs[:, None], as_tuple=False)[
                    :, 1
                ]

                edge_indices = find_edge_indices(self.G.edge_index, src_idx, dst_idx)

                edge_mask = torch.ones(self.G.edge_index.size(1), dtype=torch.bool)
                edge_mask[edge_indices] = False

                self.G.edge_index = self.G.edge_index[:, edge_mask]

                self.G.edge_attr = self.G.edge_attr[edge_mask]

                # remove isolated nodes
                edge_index, _, mask = remove_isolated_nodes(
                    self.G.edge_index, num_nodes=self.G.x.size(0)
                )
                # update edge index
                self.G.edge_index = edge_index
                self.G.x = self.G.x[mask]
                self.G.idx = self.G.idx[mask]


        # if only deletion, no need to return anything
        if not updated:
            return None

        return self.G.x, self.G.edge_index, self.G.edge_attr