import gc
from typing import Any, Dict, List

from ..save_mixin.pickle import PickleSaveMixin
from ..components.pipeline_component import PipelineComponent
from ..components.splitter import SplitterComponent
from collections import deque

from abc import abstractmethod
import numpy as np
import torch
import copy
from numpy.typing import NDArray
from ..utils import quantiles


import importlib

class BaseOnlineODModel(PickleSaveMixin, PipelineComponent):
    def __init__(
        self,
        model_str,
        queue_len=10000,
        percentile=0.95,
        warmup=1000,
        quantile_func="log_normal_quantile",
        **kwargs,
    ):
        """base interface for online outlier detection model

        Args:
            queue_len (int, optional): length of loss queue. Defaults to 10000.
            preprocessors (list, optional): list of preprocessors as strings. Defaults to [].
            profile (bool, optional): whether to profile the model, only available for pytorch models. Defaults to False.
            load_existing (bool, optional): whether to loading existing model. Defaults to False.
        """
        super().__init__(component_type="models", **kwargs)
        
        
        self.loss_queue = deque(maxlen=queue_len)
        
        self.warmup = warmup
        self.percentile=percentile 
        self.quantile_func=getattr(quantiles, quantile_func)
                
        self.batch_trained = 0
        self.batch_evaluated = 0
        
        self.module_path, self.class_name = model_str.rsplit(".", 1)
        module = importlib.import_module(self.module_path)
        self.model_cls=getattr(module, self.class_name)
        
    def setup(self):
        super().setup()
        self.model=self.model_cls(self.context)
        

    def process(self, X):
        threshold=self.get_threshold()
        # calculate score
        score = self.model.predict_step(X)

        self.batch_evaluated+=1
        
        # train if mostly benign or less than warmup
        if self.batch_trained < self.warmup or threshold > np.median(score):
            # store score for trained samples only
            if score is not None:
                self.loss_queue.extend(score)
             
            self.model.train_step(X)
            self.update_scaler()
            self.batch_trained+=1
            
        return {"threshold": threshold, "score": score, "batch_num": self.batch_evaluated}

    def update_scaler(self):
        scaler=self.context.get("scaler",None)
        if scaler:
            scaler.update_current()
    
    def get_threshold(self) -> float:
        """gets the current threshold value based on previous recorded loss value. By default it is 95th percentile

        Returns:
            float: threshold value
        """
        if len(self.loss_queue) > self.warmup:
            threshold = self.quantile_func(self.loss_queue, self.percentile)
        else:
            threshold = np.inf
        return threshold

    
    def __str__(self):
        return f"OnlineOD({self.class_name})"


class MultilayerSplitter(SplitterComponent):
    def __init__(self, **kwargs):
        """splits the input into different layers based on protocol. each layer is attached to a new pipeline"""
        super().__init__(component_type="models", **kwargs)
        self.name = f"MultlayerSplitter({self.pipeline})"

    def setup(self):
        self.parent.context["model_name"] = self.name

        if not self.loaded_from_file:
            

            for proto in self.context["protocols"].keys():
                self.pipelines[proto] = copy.deepcopy(self.pipeline)
                self.pipelines[proto].context["protocol"] = proto
                self.pipelines[proto].parent = self
                self.pipelines[proto].setup()

    def __str__(self):
        return self.name

    def split_function(self, data) -> Dict[str, Any]:
        """splits the input data

        Args:
            data (_type_): the input data

        Returns:
            Dict[str, Any]: dictionary of splits
        """
        all_results = {}
        
        for proto_name, proto_id in self.context["protocols"].items():
            selected = data[data[:, 3] == proto_id]
            if selected.size > 0:
                all_results[proto_name] = data[data[:, 3] == proto_id]

        return all_results
