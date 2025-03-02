import json
from .base_buffer import BaseFeatureBuffer
from ..save_mixin.null import NullSaveMixin
import networkx as nx


class GraphFeatureBuffer(NullSaveMixin, BaseFeatureBuffer):            
    def save_buffer(self):
        """saves buffer        
        """        
        for graph in self.feature_list:
            for root, tree in graph.items():
                
                graph_json=json.dumps(tree, default=nx.node_link_data)
                
                self.feature_file.write(graph_json)
                self.feature_file.write("\n"+"-"*50+"\n")
            self.feature_file.write("\n"+"*"*50+"\n")
        
        return self.feature_list
    
    def __str__(self):
        return f"GraphFeatureBuffer({self.buffer_size})"

