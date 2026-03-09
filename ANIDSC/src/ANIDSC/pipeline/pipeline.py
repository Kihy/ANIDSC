import importlib
import sys
import time
from typing import Dict
import os
from tqdm import tqdm
import yaml
from ..utils.helper import load_yaml

from ..component.pipeline_component import PipelineComponent
from ..save_mixin.yaml import YamlSaveMixin


class Pipeline(YamlSaveMixin, PipelineComponent):

    @property
    def pipeline_name(self):
        return self._name
    
    @pipeline_name.setter
    def pipeline_name(self, value):
        self._name = value

    def load_component(self, comp):
        module = importlib.import_module(f"ANIDSC.{comp['type']}")
        component_cls = getattr(module, comp["class"])
        if comp.get("file", False):
            file_path = comp["file"]
            comp = component_cls.load(file_path)
        else:
            comp = component_cls(**comp.get("attrs", {}))
        return comp

    def load_components(self, manifest):
        if isinstance(manifest, str):
            manifest = load_yaml(manifest)
        
        if isinstance(manifest, list):
            components = []
            for comp in manifest:
                components.append(self.load_component(comp))
        else:
            components = self.load_component(manifest)
        return components

    @property
    def components(self):
        return self._components

    def __init__(self, name, components, run_identifier):
        super().__init__()
        self._components = self.load_components(components)
        self.run_identifier = run_identifier
        self._name = name

    def setup(self):
        for i, comp in enumerate(self.components):
            comp.parent_pipeline = self
            comp.index = i
            comp.setup()

    def teardown(self):
        for i, comp in enumerate(self.components):
            comp.teardown()

    def process(self, data=None):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.start_time = time.time()
        for component in self.components:
            data = component.process(data)

            if data is None:
                break
        return data

    def start(self):
        
        pbar = tqdm(
            mininterval=float(os.getenv("TQDM_MININTERVAL", 60)),
            file=sys.stderr,
            dynamic_ncols=False,
            ascii=True,
        )

        try:
            while True:
                self.process()
                pbar.update(1)
        except StopIteration as e:
            print("Pipeline Finished")
        finally:
            self.teardown()
            self.save()

    @property
    def config_attr(self):
        attribute = super().config_attr
        # turn components into dicts for yaml serialization
        del attribute["components"]
        attribute["components"] = [v.to_dict() for v in self.components]
        return attribute
    
    @property
    def information_dict(self):
        attrs = ["dataset_name", "file_name", "pipeline_name", "run_identifier"]
        return_dict = {attr: self.search_attr(0, attr) for attr in attrs}
        return_dict["result_path"] = self.search_attr(0, "feature_path")
        
        return return_dict

    def search_attr(self, index, attr, default=None):
        """searches for attribute in the following order:
        - self
        - backward search in components
        - parent pipeline
        - forward search in components
        """
        
        # check self first
        if hasattr(self, attr):
            return getattr(self, attr)
        
        # backward search
        for i in range(index - 1, -1, -1):
            if hasattr(self.components[i], attr):
                return getattr(self.components[i], attr)

        # parent pipeline
        if self.parent_pipeline is not None:
            result = self.parent_pipeline.search_attr(self.index, attr, default)
            if result is not None:
                return result

        # forward search
        for i in range(index, len(self.components)):
            if hasattr(self.components[i], attr):
                return getattr(self.components[i], attr)

        return default

    def get_attr_by_name(self, comp_name, attr, default=None):
        for comp in self.components:  # backward search
            if comp_name == comp.name:

                if hasattr(comp, attr):
                    return getattr(comp, attr)
        return default

    def __eq__(self, other: "Pipeline"):
        same_class = self.__class__ == other.__class__
        if not same_class:
            return False

        return self.components == other.components

    def __str__(self):
        return "->".join([component.name for component in self.components])
