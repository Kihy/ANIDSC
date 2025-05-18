
from ..save_mixin.pickle import PickleSaveMixin
from ..component.pipeline_component import PipelineComponent
from collections import deque

from abc import abstractmethod
import numpy as np
from ..utils import threshold_func


import importlib

class BaseOnlineODModel(PickleSaveMixin, PipelineComponent):
    def __init__(
        self,
        model_name,
        queue_len=10000,
        percentile=0.95,
        warmup=1000,
        t_func="log_normal_quantile",
        **kwargs,
    ):
        """base interface for online outlier detection model

        Args:
            queue_len (int, optional): length of loss queue. Defaults to 10000.
            preprocessors (list, optional): list of preprocessors as strings. Defaults to [].
            profile (bool, optional): whether to profile the model, only available for pytorch models. Defaults to False.
            load_existing (bool, optional): whether to loading existing model. Defaults to False.
        """
        super().__init__(component_type="model", **kwargs)
        
        self.model_name=model_name
        self.loss_queue = deque(maxlen=queue_len)
        
        self.warmup = warmup
        self.percentile=percentile 
        self.t_func=getattr(threshold_func, t_func)
                
        self.batch_trained = 0
        self.batch_evaluated = 0
        
        if "." in model_name:
            sub_module, class_name = model_name.rsplit(".", 1)
            module_path=f"ANIDSC.model.{sub_module}"
        else:
            module_path="ANIDSC.model"
            class_name=model_name
        module = importlib.import_module(module_path)
        self.model_cls=getattr(module, class_name)
        
    def setup(self):
        super().setup()
        # request from graph_rep first
        ndim=self.request_attr("graph_rep","n_features", None)
        if not ndim:
            ndim=self.request_attr("data_source","ndim")
        self.model=self.model_cls(ndim)
        

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
        self.request_action("scaler","update_current")
    
    def get_threshold(self) -> float:
        """gets the current threshold value based on previous recorded loss value. By default it is 95th percentile

        Returns:
            float: threshold value
        """
        if len(self.loss_queue) > self.warmup:
            threshold = self.t_func(self.loss_queue, self.percentile)
        else:
            threshold = np.inf
        return threshold

    
    def __str__(self):
        return f"OnlineOD({self.model_name})"

