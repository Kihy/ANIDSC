from abc import ABC, abstractmethod
from tqdm import tqdm


from .pipeline_component import PipelineComponent




class PipelineSource(PipelineComponent):

    def __init__(
        self,
        dataset_name: str,
        file_name: str,
        max_records=float("inf"),
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.file_name = file_name
        self.max_records = max_records

        self.count = 0
        self._iter=None

    def setup(self):
        pass

    def teardown(self):
        pass
    
    @property
    @abstractmethod
    def output_dim(self):
        pass

    def process(self, _: None):

        # return None if end of iter
        if self.max_records == self.count:
            raise StopIteration("Max records reached")

        data = next(self.iter)
        if data is not None:
            self.timestamp = self.get_timestamp(data)
            
        self.count += self.batch_size
        return data

    @abstractmethod
    def get_timestamp(self, data):
        pass

    @property
    @abstractmethod
    def batch_size(self):
        pass

    @property
    def iter(self):
        return self._iter

