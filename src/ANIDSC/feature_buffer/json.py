import json
from pathlib import Path
from ..component.feature_buffer import BaseFeatureBuffer
from ..save_mixin.null import NullSaveMixin


class JsonFeatureBuffer(NullSaveMixin, BaseFeatureBuffer):
    def __init__(self, **kwargs):
        super().__init__("ndjson", write_header=False, **kwargs)

    def save_buffer(self) -> str:
        """saves buffer"""

        if len(self.data_list) == 0:
            return

        for obj in self.data_list:
            json.dump(obj, self.save_file)
            self.save_file.write("\n")

        self.data_list = []
