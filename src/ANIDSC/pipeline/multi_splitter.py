
from typing import Any, Dict
import copy

from .pipeline import Pipeline

class MultilayerSplitter(Pipeline):
    def __init__(self, **kwargs):
        """splits the input into different layers based on protocol. each layer is attached to a new pipeline"""
        super().__init__(**kwargs)

    def setup(self):
        self.components={}
        # copy pipeline and attach to each protocol layer
        for proto, pipeline in self.manifest.items():
            components=Pipeline.load(pipeline)
            components.parent_pipeline=self
            components.setup()
            components.add_prefix(proto)
            self.components[proto] = components
            
        
 
    
    def on_load(self):
        self.components={}
        # copy pipeline and attach to each protocol layer
        for proto, pipeline in self.manifest.items():
            components=Pipeline.load(pipeline)
            components.parent_pipeline=self
            components.on_load()
            components.add_prefix(proto)
            self.components[proto] = components  
        
    def process(self, data):
        split_data = self.split_function(data)
        
        for key, value in split_data.items():
            self.components[key].process(value)
        
    
    def split_function(self, data) -> Dict[str, Any]:
        """splits the input data

        Args:
            data (_type_): the input data

        Returns:
            Dict[str, Any]: dictionary of splits
        """
        all_results = {}
        
        for proto_name, proto_id in self.protocol_map.items():
            selected = data[data["protocol"] == proto_id]
            if selected.size > 0:
                all_results[proto_name] = data[data["protocol"] == proto_id]

        return all_results

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"