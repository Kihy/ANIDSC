import importlib
from pathlib import Path
from typing import Any, Dict
import inspect
import yaml
import os


class YamlSaveMixin:
    def save(self):
        # save individual components
        for key, component in self.components.items():
            component.save()

        # save manifest
        manifest_path = Path(self.get_save_path())
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as out_file:
            yaml.safe_dump(self.to_dict(), out_file, indent=4, sort_keys=False)

    @classmethod
    def load(cls, input_data):
        if isinstance(input_data, str):    
            if os.path.isfile(input_data):
                with open(input_data) as file:
                    manifest = yaml.safe_load(file)
            else:
                manifest = yaml.safe_load(input_data)

        elif isinstance(input_data, Dict):
            manifest=input_data
        else:
            raise TypeError("Unknown input_data format")
        
        
        return cls(**manifest["attrs"])
