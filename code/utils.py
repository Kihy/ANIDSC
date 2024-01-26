import json
from pathlib import Path
from io import TextIOWrapper


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
