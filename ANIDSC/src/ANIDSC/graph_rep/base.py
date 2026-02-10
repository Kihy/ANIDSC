from abc import abstractmethod
import importlib

from ..utils.helper import compare_dicts
from ..save_mixin.pickle import PickleSaveMixin

from ..component.pipeline_component import PipelineComponent
import sys

from torch_geometric.utils import from_networkx
from networkx.readwrite import json_graph


class GraphProcessor(PickleSaveMixin, PipelineComponent):
    def __init__(self, rep_name, **kwargs):
        super().__init__(**kwargs)
        self.rep_name = rep_name
        self.graph_rep = None
        

    def setup(self):
        if self.graph_rep is None:
            module = importlib.import_module("ANIDSC.graph_rep")
            self.model_cls = getattr(module, self.rep_name)
            self.graph_rep = self.model_cls()

    @property
    def output_dim(self):
        return 2

    @property
    def transformed_graph(self):
        return self._graph

    def process(self, X):
        self._graph = self.graph_rep.transform(X)

        return self._graph

    def teardown(self):
        return super().teardown()
    
    def __str__(self):
        return f"GraphProcessor({self.rep_name})"


class GraphRepresentation:
    @abstractmethod
    def transform(self, X):
        pass

    def __getstate__(self):
        return self.__dict__

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return compare_dicts(self.__getstate__(), other.__getstate__(), self.__class__)
