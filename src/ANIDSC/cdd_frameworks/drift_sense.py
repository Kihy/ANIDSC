
from collections import deque
import copy
from typing import Any, Dict, List

import numpy as np
from ANIDSC.base_files.model import BaseOnlineODModel
from ANIDSC.base_files.pipeline import PipelineComponent


class DriftSense(PipelineComponent):
    def __init__(
        self, model: BaseOnlineODModel, patience: int=1000, confidence: float=50, **kwargs
    ):
        """A wrapper around baseline model to allow concept drift detection

        Args:
            model (BaseOnlineODModel): the base model
            patience (int): patience for concept drift detection
            confidence (float): threshold to differentiate benign and malicious drift
        """
        super().__init__(component_type="models", **kwargs)

        self.model = model
        self.model_pool = []
        self.patience = patience
        self.confidence = confidence

        self.loss_queue = deque(maxlen=patience)
        self.potential_queue = deque(maxlen=patience)
        self.potential_x_queue = deque(maxlen=patience)

        self.context = {}

        self.model_idx = 0
        self.max_model_pool_size = 20
        self.time_since_last_drift = 0
        self.num_trained = 0
        self.name = self.__str__()

    def create_model(self) -> BaseOnlineODModel:
        """creates a new model based on base model

        Returns:
            BaseOnlineODModel: the new model
        """
        new_model = copy.deepcopy(self.model)
        new_model.parent = self
        new_model.setup()
        return new_model

    def setup(self):
        if not self.loaded_from_file:
            self.model_pool.append(self.create_model())
        self.parent.context["concept_drift_detection"] = True
        self.parent.context["model_name"] = f"CDD({self.model.name})"

    def process(self, data) -> Dict[str, Any]:
        """process data for concept drift detection

        Args:
            data (_type_): input data

        Returns:
            Dict[str, Any]: results of input
        """
        start_idx = self.model_idx
        trained=False
        while True:
            score, threshold = self.model_pool[self.model_idx].predict_step(
                data, preprocess=True
            )

            # found one in model pool
            if (
                self.model_pool[self.model_idx].num_trained
                < self.model_pool[self.model_idx].warmup
                or np.median(score) < threshold
            ):
                if score is not None:
                    self.model_pool[self.model_idx].loss_queue.extend(score)
                trained=True
                self.model_pool[self.model_idx].train_step(data, preprocess=True)
                self.model_pool[self.model_idx].last_batch_num = self.num_trained
                break

            self.model_idx += 1
            self.model_idx %= len(self.model_pool)

            # if cannot find one
            if self.model_idx == start_idx:
                break

        self.num_trained += 1

        if score is not None:
            self.update_queue(score, threshold, data)

        # check concept drift
        if (
            len(self.loss_queue) == self.patience
            and len(self.potential_queue) == self.patience
        ):
            difference = np.median(self.potential_queue) - np.median(self.loss_queue)
            as_range = max(
                np.percentile(self.potential_queue, 99),
                np.percentile(self.loss_queue, 99),
            ) - min(
                np.percentile(self.potential_queue, 1),
                np.percentile(self.loss_queue, 1),
            )
            diff_magnitude = (self.patience * difference**2) / (as_range**2)

            if diff_magnitude > self.confidence:

                self.clear_potential_queue()
            else:

                if len(self.model_pool) == self.max_model_pool_size:

                    # remove last unused from model pool
                    lowest_idx = 0
                    lowest_batch_num = np.inf
                    for i in range(self.max_model_pool_size):
                        if (
                            self.model_pool[self.model_idx].last_batch_num
                            < lowest_batch_num
                        ):
                            lowest_idx = i
                            lowest_batch_num = self.model_pool[
                                self.model_idx
                            ].last_batch_num

                    del self.model_pool[lowest_idx]

                new_model = self.create_model()
                # train model on existing features

                for old_x in self.potential_x_queue:
                    new_model.train_step(old_x, preprocess=True)
                trained=True
                score, threshold = new_model.predict_step(data, preprocess=True)
                new_model.loss_queue.extend(score)

                self.model_idx = len(self.model_pool)
                self.model_pool.append(new_model)

                self.clear_potential_queue()
                self.update_queue(score, threshold, data)
        else:
            diff_magnitude=0
        if score is not None:
            self.loss_queue.extend(score)

        return {
            "drift_level": diff_magnitude,
            "threshold": threshold,
            "score": score,
            "batch_num": self.num_trained,
            "model_batch_num": self.model_pool[self.model_idx].num_trained,
            "model_idx": self.model_idx,
            "num_model": len(self.model_pool),
            "trained":trained
        }


    def update_queue(self, score: List[float], threshold: float, x: Any):
        """updates the queues to check for concept drift

        Args:
            score (List[float]): list of scores from a batch
            threshold (float): threshold value
            x (Any): input data
        """
        if np.median(score) > threshold:
            self.potential_queue.append(np.median(score))
            self.potential_x_queue.append(x)
            self.time_since_last_drift = 0
        else:
            self.loss_queue.append(np.median(score))
            self.time_since_last_drift += 1

        if self.time_since_last_drift > self.patience // 5:
            self.clear_potential_queue()

    def clear_potential_queue(self):
        """clears the potential queue"""
        self.potential_queue.clear()
        self.potential_x_queue.clear()
        
    def __str__(self):
        return f"DriftSense({self.model})"