from abc import ABC
from tqdm import tqdm


from .pipeline_component import PipelineComponent


class PipelineSource(PipelineComponent):

    def __init__(self, dataset_name: str, file_name: str, max_records=float("inf"), batch_size=256):
        super().__init__(component_type="data_source")
        
        self.comparable=False

        self.dataset_name = dataset_name
        self.file_name = file_name
        self.max_records = max_records
        self.batch_size=batch_size        
        self.count = 0
        self.iter = None
        
        self.unpickleable.append('iter')
        
        self.save_attr.extend(['dataset_name','file_name','max_records','batch_size'])
        
        
    def setup(self):
        pass

    def process(self, _:None):
        
        # return None if end of iter
        if self.max_records==self.count:
            return None

        data=next(self.iter, None)
        self.count += self.batch_size
        return data 
        
