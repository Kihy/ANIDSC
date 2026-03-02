from ..save_mixin.pickle import PickleSaveMixin

from ..component.pipeline_component import PipelineComponent


class RemoveTimestamp(PickleSaveMixin, PipelineComponent):

    def process(self, data):
        return data.drop(columns=["timestamp"])
    
    def setup(self):
        pass

    def teardown(self):
        pass

    @property
    def output_dim(self):
        return self.request_attr("output_dim") - 1