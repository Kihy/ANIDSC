from typing import List
from ..graph_rep.base import GraphRepresentation
import networkx as nx 

class Barebone(GraphRepresentation):
    
    def transform(self, X: List[nx.Graph]):
        """ strips attributes away from graph
        """
        
        
        transformed=[]
        for g in X:
            G = nx.Graph()
            G.add_nodes_from(g.nodes())
            G.add_edges_from(g.edges())
            transformed.append(G)

        return transformed