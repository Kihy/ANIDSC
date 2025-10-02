import importlib
import time
from typing import Dict

from tqdm import tqdm
import yaml

from ..save_mixin.pickle import PickleSaveMixin
from ..component.pipeline_component import PipelineComponent
from ..save_mixin.yaml import YamlSaveMixin


class Pipeline(YamlSaveMixin, PipelineComponent):

    @property
    def components(self):
        return self._components

    def __init__(self, components):
        super().__init__()

        self.start_time = None
        self._components=components
        

    def setup(self):
        for i,comp in enumerate(self.components):
            comp.parent_pipeline = self
            comp.index=i 
            comp.setup()      

    def process(self, data=None):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.start_time = time.time()
        for  component in self.components:
            data = component.process(data)

            if data is None:
                break
       

    def start(self):

        pbar = tqdm()

        try:
            while True:
                self.process()
                pbar.update(1)
        except StopIteration as e:
            print("Pipeline Finished")
        finally:
            self.save()

    @property 
    def config_attr(self):
        return {"manifest":[v.to_dict() for v in self.components]}
        
       
    def get_attr(self, index, attr, default=None):
        for i in range(index, -1, -1):  # backward search
            comp = self.components[i]

            if hasattr(comp, attr):
                return getattr(comp, attr)
        
        for i in range(index+1, len(self.components)):  # forwards search
            comp = self.components[i]

            if hasattr(comp, attr):
                return getattr(comp, attr)

        # check self
        if hasattr(self, attr):
            return getattr(self, attr)
        
        return default

    def __eq__(self, other: "Pipeline"):
        same_class = self.__class__ == other.__class__
        if not same_class:
            return False

        return self.components == other.components

    def __str__(self):
        return "->".join([str(component) for component in self.components])
