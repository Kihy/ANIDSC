from .pipeline import PipelineComponent
import copy 

class BaseOnlineNormalizer(PipelineComponent):
    def __init__(self, warmup=10, ndim=0, skip=0, **kwargs):
        super().__init__(component_type="scalers",**kwargs)
        self.warmup=warmup
        self.ndim=ndim 
        self.skip=skip
        
    def setup(self):
        context=self.get_context()
        self.ndim=context['fe_features']
        self.skip=context['skip']

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result