import math

import torch
from ..save_mixin.pickle import PickleSaveMixin
from ..component.pipeline_component import PipelineComponent
from collections import deque

from abc import abstractmethod
import numpy as np
from ..utils import threshold_func
from ..scaler import LivePercentile

import importlib


class BaseOnlineODModel(PickleSaveMixin, PipelineComponent):
    def __init__(
        self,
        model_name,
        queue_len=10000,
        percentile=0.99,
        warmup=1000,
        t_func="quantile",
        **kwargs,
    ):
        """base interface for online outlier detection model

        Args:
            queue_len (int, optional): length of loss queue. Defaults to 10000.
            preprocessors (list, optional): list of preprocessors as strings. Defaults to [].
            profile (bool, optional): whether to profile the model, only available for pytorch models. Defaults to False.
            load_existing (bool, optional): whether to loading existing model. Defaults to False.
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.loss_queue = deque(maxlen=queue_len)

        self.warmup = warmup
        self.percentile = percentile
        self.t_func = getattr(threshold_func, t_func)
        self.tolerance=2
        
        self.batch_trained = 0
        self.batch_evaluated = 0
        self.model = None

    def teardown(self):
        pass

    def setup(self):

        if self.model is None:
            if "." in self.model_name:
                sub_module, class_name = self.model_name.rsplit(".", 1)
                module_path = f"ANIDSC.model.{sub_module}"
            else:
                module_path = "ANIDSC.model"
                class_name = self.model_name

            module = importlib.import_module(module_path)
            self.model_cls = getattr(module, class_name)

            ndim = self.request_attr("output_dim")

            self.model = self.model_cls(ndim)

    def process(self, X):
        threshold = self.get_threshold()
        score = np.full((X.shape[0],), self.tolerance*threshold+1.)
        
        # Ensure X is a tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).cuda()
        
        # Get valid samples (no inf values)
        valid_mask = ~torch.isinf(X).any(dim=1)
        score[valid_mask.cpu().numpy()] = self.model.predict_step(X[valid_mask])
        self.batch_evaluated+=1
        
        
        # Determine training mask and scores to queue
        if self.batch_trained < self.warmup:
            train_mask = valid_mask    
        else:
            train_mask = valid_mask.cuda() & torch.from_numpy(score < self.tolerance*threshold).cuda()
        
        queue_scores = score[train_mask.cpu().numpy()]
        
        # Update loss queue and train
        self.loss_queue.extend(queue_scores)
        
        self.model.train_step(X[train_mask])
        self.batch_trained += 1
        
        return {"threshold": threshold, "score": score}

    def get_threshold(self) -> float:
        """gets the current threshold value based on previous recorded loss value. By default it is 99th percentile
        If its in warmup stage, the threshold is -1
        Returns:
            float: threshold value
        """
        if len(self.loss_queue) > self.warmup:
            threshold = self.t_func(self.loss_queue, self.percentile)
        else:
            threshold = -1
        return threshold

    def __str__(self):
        return f"OnlineOD({self.model_name})"
