
from typing import Any, Dict
import copy
from ..component.pipeline_component import PipelineComponent
from ..save_mixin import PickleSaveMixin
from ..templates import PipelineRegistry
from .pipeline import Pipeline
import numpy as np

class MultilayerSplitter(PickleSaveMixin, Pipeline):
    def __init__(self, name, components, run_identifier, protocol_map, **kwargs):
        """splits the input into different layers based on protocol. each layer is attached to a new pipeline"""
        super().__init__(name, components, run_identifier, **kwargs)

        self.protocol_map = protocol_map
        self.inner_pipelines = {}
        self.batch_evaluated=0
            
    @property
    def name(self):
        return f"MultilayerSplitter({self.inner_pipelines[next(iter(self.inner_pipelines))].name})"
    @property
    def components(self):
        return self.inner_pipelines
    
    
    def setup(self):
        # copy pipeline and attach to each protocol layer
        for proto_name, proto_id in self.protocol_map.items():
            inner_pipeline=copy.deepcopy(self._components)
            inner_pipeline.index=0
            inner_pipeline.parent_pipeline=self
            inner_pipeline.pipeline_name=f"{self.request_attr('pipeline_name')}/{proto_name}"
            inner_pipeline.setup()

            self.inner_pipelines[proto_name] = inner_pipeline
        
        # set parent pipeline_name to /full 
        self.parent_pipeline.pipeline_name=f"{self.request_attr('pipeline_name')}/full"

    @property
    def config_attr(self):
        pipelines={}
        for proto_name, pipeline in self.inner_pipelines.items():
            pipelines[proto_name]=pipeline.config_attr
        return pipelines
    
    def process(self, data):
        

        split_data = self.split_function(data)

        # aggregate results from each layer
        score=[]
        for key, value in split_data.items():
            
            if value.size>0:
                processed=self.inner_pipelines[key].process(value)

                if processed is None:
                    continue 
                
                # skip if threshold is negative (indicating warmup)
                if processed["threshold"]<0:
                    score.append(-1.)
                else:
                    
                    relative_score=processed["score"]/processed["threshold"]
                    
                    # take the max relative score as the score for this layer
                    score.append(np.max(relative_score))
            else:
                # skip if no data 
                score.append(-1.)
        
        score=np.array(score)
        threshold=1 if (score!=-1.).any() else -1.
        results={"threshold": threshold, "score": score}
        self.batch_evaluated += 1
        return results
    
    
    
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
            
            all_results[proto_name] = selected
            
        return all_results

    
    def teardown(self):
        for pipeline in self.inner_pipelines.values():
            pipeline.teardown()