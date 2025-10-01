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
        for key, comp in self.components.items():
            comp.setup()      
            comp.parent_pipeline = self

    def perform_action(self, comp_type, action):
        if comp_type in self.components:
            return getattr(self.components[comp_type], action)()

    def process(self, data=None):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.start_time = time.time()
        for comp_type, component in self.components.items():
            data = component.process(data)

            if data is None:
                break
        return comp_type

    def start(self):

        pbar = tqdm()

        try:
            while True:
                self.process()

                pbar.update(1)
        except StopIteration as e:
            print("End of File")
        finally:
            self.save()

   
       
    def get_attr(self, comp_type, attr, default=None):
        if comp_type in self.components:
            return getattr(self.components[comp_type], attr, default)
        elif comp_type == "":
            return getattr(self, attr, default)
        else:
            return self.request_attr(comp_type, attr, default)

    def __eq__(self, other: "Pipeline"):
        same_class = self.__class__ == other.__class__
        if not same_class:
            return False

        return self.components == other.components

    def __str__(self):
        return "->".join([str(component) for _, component in self.components.items()])
