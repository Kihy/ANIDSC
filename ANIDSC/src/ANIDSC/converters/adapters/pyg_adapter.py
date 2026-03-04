"""Adapter for NumPy arrays - AUTO-REGISTERED"""
import hashlib
from typing import Any
from ..base import AutoRegisterAdapter
import torch 
import numpy as np

try:
    import torch_geometric as pyg
    from torch_geometric.data import Data
    import networkx as nx
    
    class DataAdapter(AutoRegisterAdapter):
        """Handles all conversions TO np.ndarray"""
        
        @staticmethod
        def target_type() -> type:
            return Data
        
        @staticmethod
        def can_convert(value: Any) -> bool:
            return isinstance(value, (
               nx.Graph, nx.DiGraph
            )) 
        @staticmethod
        def convert(value: Any) -> Data:
            def _extract_features(items, key="feature"):
                features = [attrs.pop(key, None) for *_, attrs in items]
                if len(features)==0:
                    return None
                return torch.as_tensor(np.stack(features).astype(np.float32))

            def _extract_attrs(items):
                """Collect all non-feature attrs as tensors across all nodes/edges."""
                all_attrs = [attrs for *_, attrs in items]
                return {
                    key: torch.tensor([attrs[key] for attrs in all_attrs])
                    for key in all_attrs[0]
                }
            
            if isinstance(value, Data):
                return value
            elif isinstance(value, (nx.Graph, nx.DiGraph)):
                g = value.copy()  # avoid modifying original graph
                node_features = _extract_features(g.nodes(data=True))
                edge_features = _extract_features(g.edges(data=True))
                node_attrs = _extract_attrs(g.nodes(data=True))
                edge_attrs = _extract_attrs(g.edges(data=True))

                data = pyg.utils.from_networkx(g)
                
                if node_features is not None:
                    data.x=node_features 
                else:
                    data.x=torch.ones((data.num_nodes, 1))  # default feature if none provided
                    
                data.x=data.x.float()  # ensure float32 for PyG compatibility
                if edge_features is not None:
                    data.edge_attr = edge_features

                for key, val in node_attrs.items():
                    setattr(data, key, val)

                for key, val in edge_attrs.items():
                    setattr(data, key, val)

                return data.to("cuda")

        
except ImportError:
    # NumPy not available, skip this adapter
    pass