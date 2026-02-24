"""Adapter for NumPy arrays - AUTO-REGISTERED"""
import hashlib
from typing import Any
from ..base import AutoRegisterAdapter
import torch 

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
            if isinstance(value, Data):
                return value
            elif isinstance(value, (nx.Graph, nx.DiGraph)):
                # Check if edges have any attributes
                edge_attr_keys = set()
                for _, _, d in value.edges(data=True):
                    edge_attr_keys.update(d.keys())

                if not edge_attr_keys:
                    edge_attr=None 
                else:
                    edge_attr="all"
                    
                node_attr_keys = set()
                for _, d in value.nodes(data=True):
                    node_attr_keys.update(d.keys())

                if not node_attr_keys:
                    node_attr=None 
                else:
                    node_attr="all"
                
                data = pyg.utils.from_networkx(value, group_node_attrs=node_attr, group_edge_attrs=edge_attr)
                
                if data.x is not None:
                    data.x = data.x.float()
                else:
                    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
                    
                return data.to("cuda")

except ImportError:
    # NumPy not available, skip this adapter
    pass