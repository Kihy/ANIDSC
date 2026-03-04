

from typing import List, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_isolated_nodes

from .base import GraphRepresentation
import networkx as nx
from ..converters.decorator import auto_cast_method

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


class HomoGraphRepresentation(GraphRepresentation):
    def __init__(
        self,
        **kwargs,
    ):
        """Homogeneous graph representation of network

        Args:
            device (str, optional): name of device. Defaults to "cuda".
            preprocessors (List[str], optional): list of preprocessors. Defaults to [].
        """


        self.n_features = 15
        self.l_features = 35
        
        # Pre-compute column offsets
        self.offset = 4
        self.src_feat_start = self.offset
        self.src_feat_end = self.offset + self.n_features
        self.dst_feat_start = self.src_feat_end
        self.dst_feat_end = self.dst_feat_start + self.n_features
        self.edge_feat_start = self.dst_feat_end
        self.edge_feat_end = self.edge_feat_start + self.l_features
        
        # initialize NetworkX directed graph
        self.G = nx.DiGraph()
        
        # Initialize ID mapping (string -> integer)
        self.id_mapping = {}  # string_id -> int_id
        self.reverse_id_mapping = {}  # int_id -> string_id
        self.next_id = 0

    @property
    def output_dim(self):
        # number of node features 
        return self.n_features


    def teardown(self):
        pass
    
    def setup(self):
        super().setup()
        


    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        """updates source and destination nodes

        Args:
            srcID (array): integer source node IDs
            src_feature (array): source node features
            dstID (array): integer destination node IDs
            dst_feature (array): destination node features
        """
        # Update source nodes
        for i in range(len(srcID)):
            int_id = srcID[i]
            features = src_feature[i] if isinstance(src_feature[i], np.ndarray) else np.array(src_feature[i])
            self.G.add_node(int_id, feature=features, updated=True)
        
        # Update destination nodes
        for i in range(len(dstID)):
            int_id = dstID[i]
            features = dst_feature[i] if isinstance(dst_feature[i], np.ndarray) else np.array(dst_feature[i])
            self.G.add_node(int_id, feature=features, updated=True)



    def update_edges(self, srcID, dstID, edge_feature):
        """updates edges between source and destination

        Args:
            srcID (array): integer source node IDs
            dstID (array): integer destination node IDs
            edge_feature (array): edge features
        """
        for i in range(len(srcID)):
            src_int_id = srcID[i]
            dst_int_id = dstID[i]
            features = edge_feature[i] if isinstance(edge_feature[i], np.ndarray) else np.array(edge_feature[i])
            
            # Add or update edge with features
            self.G.add_edge(src_int_id, dst_int_id, feature=features)

    def get_node_id(self, str_id):
        """Convert string ID to integer ID with consistent mapping
        
        Args:
            str_id: string representation of node ID
            
        Returns:
            int: integer node ID
        """
        str_id = str(str_id)  # Ensure it's a string
        if str_id not in self.id_mapping:
            self.id_mapping[str_id] = self.next_id
            self.reverse_id_mapping[self.next_id] = str_id
            self.next_id += 1
        return self.id_mapping[str_id]

    def split_data(self, x):
        """splits input data into different parts and converts string IDs to integers

        Args:
            x (_type_): input data

        Returns:
            _type_: converted graph IDs and features
        """
        srcMAC = x[:, 1]
        dstMAC = x[:, 2]
        
        # Convert string IDs to integer IDs
        srcIDs_int = np.array([self.get_node_id(sid) for sid in srcMAC])
        dstIDs_int = np.array([self.get_node_id(did) for did in dstMAC])
        
        # cast to float 32 for PyG compatibility
        return (srcIDs_int, dstIDs_int, 
                x[:, self.src_feat_start:self.src_feat_end].astype(np.float32),
                x[:, self.dst_feat_start:self.dst_feat_end].astype(np.float32),
                x[:, self.edge_feat_start:self.edge_feat_end].astype(np.float32))



    @auto_cast_method
    def transform(self, x: np.ndarray):
        """updates self.G with x and returns the updated graph

        Args:
            x (_type_): input data features

        Returns:
            networkx.DiGraph: Updated NetworkX graph representation
        """

        # Use numpy operations to find split indices
        diff = np.diff(x[:, 0])
        split_indices = np.nonzero(diff)[0] + 1
        split_indices = np.concatenate([[0], split_indices, [len(x)]])
        split_sizes = np.diff(split_indices)
        
        updated = False

        # Split data by flow (indicated by column 0)
        current_idx = 0
        for size in split_sizes:
            data = x[current_idx:current_idx + size]
            current_idx += size

            # Add/update links
            if data[0, 0] == 1:
                srcIDs, dstIDs, src_features, dst_features, edge_features = self.split_data(data)
                
                # Find last occurrence of each unique connection
                unique_pairs, indices = np.unique(
                    np.column_stack([srcIDs, dstIDs]),
                    axis=0,
                    return_index=True
                )
                
                # Sort indices to get last occurrence
                last_indices = np.argsort(-data[indices, 0])[:len(unique_pairs)]
                indices = indices[last_indices]

                # Update nodes and edges directly on self.G
                self.update_nodes(srcIDs[indices], src_features[indices],
                                 dstIDs[indices], dst_features[indices])
                self.update_edges(srcIDs[indices], dstIDs[indices], edge_features[indices])
                updated = True

            # Delete links
            else:
                srcIDs = data[:, 1]
                dstIDs = data[:, 2]
                
                # Convert string IDs to integer IDs
                srcIDs_int = np.array([self.get_node_id(sid) for sid in srcIDs])
                dstIDs_int = np.array([self.get_node_id(did) for did in dstIDs])

                # Remove edges from self.G
                for src_int_id, dst_int_id in zip(srcIDs_int, dstIDs_int):
                    if self.G.has_edge(src_int_id, dst_int_id):
                        self.G.remove_edge(src_int_id, dst_int_id)
                
                # Remove isolated nodes
                isolated = list(nx.isolates(self.G))
                self.G.remove_nodes_from(isolated)

        # Return None if graph is empty after updates
        if nx.is_empty(self.G):
            return None
        
        return self.G