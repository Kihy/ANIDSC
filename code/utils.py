import json
from pathlib import Path
from io import TextIOWrapper

class LazyInitializationMixin:
    def lazy_init(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.allowed
            setattr(self, k, v)
            
    def start(self, **kwargs):
        
        for k, v in kwargs.items():
            assert k in self.allowed
            setattr(self, k, v)

        return self.entry()

    def __rrshift__(self, other):
        return self.start(**other)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, TextIOWrapper):
            return obj.name
        return super().default(obj)


def load_dataset_info():
    with open("data_info.json", "r") as f:
        data_info = json.load(f)
    return data_info


def save_dataset_info(data_info):
    with open("data_info.json", "w") as f:
        json.dump(data_info, f, indent=4, cls=JSONEncoder)
