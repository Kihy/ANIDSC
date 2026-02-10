from typing import List
from ..component.feature_extractor import BaseFeatureExtractor
from ..save_mixin.pickle import PickleSaveMixin

from ..converters import auto_cast_method
from ..converters.types import RecordList
import numpy as np
import networkx as nx



class GraphFeatureExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        
        self.features=["num_nodes", "num_edges", "edge_density","mean_degree",
                "degree_variance", "max_degree", "lcc_ratio"]
    def setup(self):
        pass

    def teardown(self):
        pass

    def peek(self):
        pass 

    @auto_cast_method
    def update(self, graphs: List[nx.Graph]) -> np.ndarray:
        
        return [self.update_single(g) for g in graphs]
        
    
    def update_single(self, G: nx.Graph)->List[float]:
        """updates the internal state with traffic vector

        Args:
            traffic_vector (array): a traffic vectors consists of
            data extracted from pacekts

        Returns:
            array: the extracted features
        """
        result_array=[]
        for name in self.features:
            result_array.append(getattr(self, name)(G))
        
        return result_array

    @property
    def headers(self):
        """returns the feature names

        Returns:
            list: list of feature names
        """

        return self.features
    
    def num_nodes(self,G):
        return G.number_of_nodes()

    def num_edges(self,G):
        return G.number_of_edges()
    
    def edge_density(self,G):
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        return 2 * G.number_of_edges() / (n * (n - 1))


    def mean_degree(self,G):
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        return 2 * G.number_of_edges() / n

    def degree_variance(self,G):
        degrees = [d for _, d in G.degree()]
        if len(degrees) == 0:
            return 0.0
        return np.var(degrees)

    def max_degree(self,G):
        degrees = [d for _, d in G.degree()]
        return max(degrees) if degrees else 0

    def lcc_ratio(self,G):
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
        largest_cc = max(nx.connected_components(G), key=len, default=[])
        return len(largest_cc) / n

    def avg_shortest_path_length_lcc(self,G):
        if G.number_of_nodes() == 0:
            return 0.0
        largest_cc = max(nx.connected_components(G), key=len, default=[])
        if len(largest_cc) <= 1:
            return 0.0
        H = G.subgraph(largest_cc)
        return nx.average_shortest_path_length(H)

    def diameter_lcc(self,G):
        if G.number_of_nodes() == 0:
            return 0
        largest_cc = max(nx.connected_components(G), key=len, default=[])
        if len(largest_cc) <= 1:
            return 0
        H = G.subgraph(largest_cc)
        return nx.diameter(H)