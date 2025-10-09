import json
from pathlib import Path
from ..component.feature_buffer import BaseFeatureBuffer
from ..save_mixin.pickle import PickleSaveMixin


class JsonFeatureBuffer(PickleSaveMixin, BaseFeatureBuffer):
    
    @property 
    def file_type(self):
        return "ndjson"
    
    @property
    def buffer_size(self):
        return 1
    
    def save_buffer(self) -> str:
        """saves buffer"""

   

        flat = [item for sublist in self.data_list for item in (sublist if isinstance(sublist, list) else [sublist])]

        for g in flat:
            json.dump(g, self.save_file)
            self.save_file.write("\n")

            self.data_list=[]
        return flat 
