from .pipeline import PipelineComponent

class BaseOnlineNormalizer(PipelineComponent):
    def __init__(self, warmup=10, ndim=0, skip=0):
        super().__init__()
        self.warmup=warmup
        self.ndim=ndim 
        self.skip=skip
        
    def setup(self):
        context=self.get_context()
        self.ndim=context['n_features']
        self.skip=context['skip']

    