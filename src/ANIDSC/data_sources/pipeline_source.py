from abc import ABC
from tqdm import tqdm
from ..components.pipeline_component import Pipeline


class PipelineSource(ABC):
    """The data source of pipeline. This is attached to pipeline with >>"""

    def __init__(self, dataset_name: str, file_name: str, max_records=float("inf"), batch_size=256):
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.max_records = max_records
        self.batch_size=batch_size
        self.context = {"dataset_name":dataset_name,
                        "file_name":file_name,
                        "batch_size":batch_size}
        
        self.count = 0
        self.other = None
        self.iter = None

    def on_start(self):
        self.context["pipeline_name"]=str(self.other)
        self.other.setup()
        

    
    def on_end(self):
        return self.other.teardown()


    def start(self):
        self.on_start()

        for data in tqdm(self.iter):
            self.other.process(data)
            self.count += self.batch_size

            if self.count == self.max_records:
                break

        self.on_end()

    def __rshift__(self, other: Pipeline):
        """attaches itself with a pipeline

        Args:
            other (Pipeline): the pipeline to feed data to
        """
        self.other = other
        other.set_context(self.context)

