
from .pipeline import PipelineComponent, SplitterComponent
from collections import deque
from ..utils import LivePercentile, calc_quantile
from abc import abstractmethod 
import numpy as np 
import torch
import copy

class BaseOnlineODModel(PipelineComponent):
    def __init__(self, queue_len=10000, preprocessors=[], profile=False, load_existing=False, **kwargs):
        super().__init__(component_type='models', **kwargs)
        self.preprocessors=preprocessors
        self.loss_queue = deque(maxlen=queue_len)
        
        self.num_batch=0
        self.converged = False
        self.profile=profile
        self.load_existing=load_existing
        self.suffix=[]
        
        self.custom_params=["num_batch", "loss_queue"]

    def setup(self):
        
        self.parent.context['model_name']=self.name
        context=self.get_context()
        self.suffix.append(context.get('protocol',None))
        
        if self.profile:
            log_dir=f"{context['dataset_name']}/runs/{context['pipeline_name']}"
            
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=2, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    log_dir),
                record_shapes=False,
                with_stack=False,
            )
            self.prof.start()
    
        if self.load_existing:
            self.load()
        
    @abstractmethod
    def process(self, X):
        pass 
    
    def train_step(self, X, preprocess=False):
        if preprocess:
            X=self.preprocess(X)
        self.optimizer.zero_grad()
        _, loss=self.forward(X, include_dist=True)
        loss.backward()
        self.optimizer.step()
        self.num_batch+=1
        if self.profile:
            self.prof.step()
        return loss.detach().cpu().item()
    
    @abstractmethod
    def predict_step(self, X):
        pass 

    def get_threshold(self):
        if len(self.loss_queue)>100:
            threshold=calc_quantile(self.loss_queue, 0.95)
        else:
            threshold = np.inf
        return threshold

    def preprocess(self, X):
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = getattr(self, p)(X)
        return X
    
    def to_device(self, X):
        return X.to(self.device)
    
    def to_float_tensor(self, X):
        return torch.from_numpy(X).float()
    
    def teardown(self):
        suffix='-'.join([s for s in self.suffix if s is not None])
        self.save(suffix) 
    

class ConceptDriftWrapper(PipelineComponent):
    """A component that splits the input data into different outputs and attaches each output to separate pipelines."""

    def __init__(self, model:BaseOnlineODModel,
                 patience:int, confidence:float, **kwargs):
        super().__init__(component_type="models",**kwargs)
        self.model = model
        self.model_pool=[]
        self.patience=patience
        self.confidence=confidence
        
        self.loss_queue = deque(maxlen=patience)
        self.potential_queue = deque(maxlen=patience)
        self.potential_x_queue = deque(maxlen=patience)
        
        self.context={}
        
        self.model_idx = 0
        self.max_model_pool_size=20
        self.time_since_last_drift=0
        self.num_batch=0

    def set_context(self, context):
        self.context=context
    
    def create_model(self):
        new_model=copy.deepcopy(self.model) 
        new_model.parent=self
        new_model.setup()
        return new_model
    
    def setup(self):
        self.model_pool.append(self.create_model())
        self.parent.context["concept_drift_detection"]=True
        self.parent.context['model_name']=f"CDD({self.model.name})"
 
    def process(self, data):
        start_idx=self.model_idx
        
        while True:
            score, threshold = self.model_pool[self.model_idx].predict_step(data, preprocess=True)
            
            # found one in model pool
            if np.median(score)<threshold:
                self.model_pool[self.model_idx].train_step(data, preprocess=True)
                if (score!=0).all():
                    self.model_pool[self.model_idx].loss_queue.extend(score)
                self.model_pool[self.model_idx].last_batch_num=self.num_batch
                break

            
            self.model_idx+=1
            self.model_idx%=len(self.model_pool)
            
            # if cannot find one
            if self.model_idx==start_idx:
                break
        
        self.num_batch+=1
        
        self.update_queue(score, threshold, data)
        
        #check concept drift 
        if (
            len(self.loss_queue) == self.patience
            and len(self.potential_queue) == self.patience
        ):
            difference = np.median(self.potential_queue) - np.median(self.loss_queue)
            as_range = max(np.percentile(self.potential_queue,99), np.percentile(self.loss_queue,99)) - min(
                np.percentile(self.potential_queue,1), np.percentile(self.loss_queue,1)
            )
            diff_magnitude = (self.patience * difference**2) / (as_range**2)

            if diff_magnitude > self.confidence:
                drift_level = "malicious"
                self.clear_potential_queue()
            else:
                drift_level = "benign"
                if len(self.model_pool)==self.max_model_pool_size:
                    
                    #remove last unused from model pool
                    lowest_idx=0
                    lowest_batch_num=np.inf
                    for i in range(self.max_model_pool_size):
                        if self.model_pool[self.model_idx].last_batch_num< lowest_batch_num:
                            lowest_idx=i 
                            lowest_batch_num=self.model_pool[self.model_idx].last_batch_num
                            
                    del self.model_pool[lowest_idx]
                
                new_model = self.create_model()
                # train model on existing features
            
                for old_x in self.potential_x_queue:
                    new_model.train_step(old_x, preprocess=True)
                
                score, threshold = new_model.predict_step(data, preprocess=True)
                new_model.loss_queue.extend(score)
                                
                self.model_idx = len(self.model_pool)
                self.model_pool.append(new_model)
                
                self.clear_potential_queue()
                self.update_queue(score, threshold, data)
                

        elif len(self.loss_queue) != self.patience:
            drift_level = "unfull loss queue"
        else:
            drift_level = "no drift"

        return {
            "drift_level": drift_level,
            "threshold": threshold,
            "score": score,
            "batch_num": self.num_batch,
            "model_batch_num":self.model_pool[self.model_idx].num_batch,
            "model_idx": self.model_idx,
            "num_model": len(self.model_pool),
        }
    
    def __str__(self):
        return f"ConceptDriftWrapper({self.model})"
    
    def update_queue(self, score, threshold, x):
        if np.median(score)>threshold:
            self.potential_queue.append(np.median(score))
            self.potential_x_queue.append(x)
            self.time_since_last_drift=0
        else:
            self.loss_queue.append(np.median(score))
            self.time_since_last_drift+=1
        
        if self.time_since_last_drift>self.patience//5:
            self.clear_potential_queue()
            
    def clear_potential_queue(self):
        self.potential_queue.clear()
        self.potential_x_queue.clear()

    # def teardown(self):
    #     for i, model in enumerate(self.model_pool):
    #         model.suffix.append(str(i))
    #         model.teardown()


class MultilayerSplitter(SplitterComponent):
    def __init__(self, **kwargs):
        super().__init__(component_type="models", **kwargs)
        self.name=f"{self.__class__.__name__}({self.pipeline})"
        
        
    def setup(self):
        context=self.get_context()
        for proto in context['protocols'].keys():
            self.pipelines[proto]=copy.deepcopy(self.pipeline)    
            self.pipelines[proto].context['protocol']=proto
            self.pipelines[proto].parent=self 
            self.pipelines[proto].setup()
        
        self.parent.context['model_name']=self.name
        
    def split_function(self, data):
        all_results = {}
        context=self.get_context()
        for proto_name, proto_id in context['protocols'].items():
            selected=data[data[:, 3] == proto_id]
            if selected.size >0:
                all_results[proto_name] = data[data[:, 3] == proto_id]
            
        return all_results