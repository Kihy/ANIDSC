import gc
from typing import Any, Dict, List
from .pipeline import PipelineComponent, SplitterComponent
from collections import deque
from ..utils import calc_quantile
from abc import abstractmethod
import numpy as np
import torch
import copy
from numpy.typing import NDArray


class BaseOnlineODModel(PipelineComponent):
    def __init__(
        self,
        queue_len=10000,
        preprocessors: List[str] = [],
        profile: bool = False,
        device="cuda",
        loss_dist="lognorm",
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
        self.preprocessors = [getattr(self, p) for p in preprocessors]
        self.loss_queue = deque(maxlen=queue_len)
        self.loss_dist = loss_dist

        self.warmup = 1000
        self.device = device
        self.num_trained = 0
        self.num_evaluated = 0
        self.converged = False
        self.profile = profile

        self.suffix = []

    def setup(self):
        self.parent.context["model_name"] = self.name
        context = self.get_context()
        self.suffix.append(context.get("protocol", None))

        if self.profile:
            log_dir = f"{context['dataset_name']}/runs/{context['pipeline_name']}"

            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                record_shapes=False,
                with_stack=False,
            )
            self.prof.start()

        if not self.loaded_from_file:
            self.init_model(context)

        print(f"{self.__str__()} has {self.get_total_params()} params")

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def init_model(self, context):
        pass

    def process(self, X):
        X_scaled = self.preprocess(X)

        score, threshold = self.predict_step(X_scaled)

        trained=False 
        # do not train if malicious or less than warmup
        if self.num_trained < self.warmup or threshold > np.median(score):
            if score is not None:
                self.loss_queue.extend(score)
            self.train_step(X_scaled)
            trained=True

        return {"threshold": threshold, "score": score, "batch_num": self.num_evaluated,
                "trained":trained}

    def predict_step(self, X, preprocess=False):
        if preprocess:
            X = self.preprocess(X)
        _, loss = self.forward(X, inference=True)
        
        if loss is None:
            return None, None
        
        self.num_evaluated += 1
        return loss.detach().cpu().numpy(), self.get_threshold()

    def get_loss(self, X, preprocess: bool = False):
        if preprocess:
            X = self.preprocess(X)
        _, loss = self.forward(X, inference=False)
        if loss is not None:
            loss = loss.mean()
            if self.profile:
                self.prof.step()
        return loss
    
    

    def train_step(self, X, preprocess: bool = False) -> NDArray:
        """train the model on batch X

        Args:
            X (_type_): input data to the model
            preprocess (bool, optional): whether to preprocess the data. Defaults to False.

        Returns:
            _type_: _description_
        """
        if self.optimizer is None:
            self.forward(X, inference=False)
            
            return 0.
        
        if preprocess:
            X = self.preprocess(X)
        self.optimizer.zero_grad()
        _, loss = self.forward(X, inference=False)
        
        if loss is not None:
            loss = loss.mean()
            # retain for node encoder
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.num_trained += 1

            if self.profile:
                self.prof.step()
        else:
            loss=0.
            
        # update scaler
        context=self.get_context()
        if "scaler" in context.keys():
            context["scaler"].update_current()
        
        return loss.detach().cpu().item()

    @abstractmethod
    def forward(self, X):
        """forward pass of model, returns the output and loss

        Args:
            X (_type_): _description_
        """
        pass

    def get_threshold(self) -> float:
        """gets the current threshold value based on previous recorded loss value. By default it is 95th percentile

        Returns:
            float: threshold value
        """
        if len(self.loss_queue) > 100:
            threshold = calc_quantile(self.loss_queue, 0.95, self.loss_dist)
        else:
            threshold = np.inf
        return threshold

    def preprocess(self, X):
        """preprocesses the input with preprocessor

        Args:
            X (_type_): input data

        Returns:
            _type_: preprocessed X
        """
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = p(X)
        return X

    def to_device(self, X: torch.Tensor) -> torch.Tensor:
        """preprocessor that converts X to particular device

        Args:
            X (torch.Tensor): the input data

        Returns:
            torch.Tensor: output tensor
        """
        return X.to(self.device)
    
    def to_numpy(self, X: torch.Tensor)->NDArray:
        return X.detach().cpu().numpy()

    def to_float_tensor(self, X: NDArray) -> torch.Tensor:
        """converts input data to pytorch tensor

        Args:
            X (NDArray): input data

        Returns:
            torch.Tensor: output
        """
        return torch.from_numpy(X).float()

    def teardown(self):
        if self.save:
            suffix = "-".join([s for s in self.suffix if s is not None])
            self.save_pickle(self.component_type, suffix)
            
        # clean up memory
        gc.collect()
        torch.cuda.empty_cache()





class MultilayerSplitter(SplitterComponent):
    def __init__(self, **kwargs):
        """splits the input into different layers based on protocol. each layer is attached to a new pipeline"""
        super().__init__(component_type="models", **kwargs)
        self.name = f"MultlayerSplitter({self.pipeline})"

    def setup(self):
        self.parent.context["model_name"] = self.name

        if not self.loaded_from_file:
            context = self.get_context()

            for proto in context["protocols"].keys():
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
        context = self.get_context()
        for proto_name, proto_id in context["protocols"].items():
            selected = data[data[:, 3] == proto_id]
            if selected.size > 0:
                all_results[proto_name] = data[data[:, 3] == proto_id]

        return all_results
