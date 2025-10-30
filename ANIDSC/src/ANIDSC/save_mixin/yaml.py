import importlib
from pathlib import Path
from typing import Any, Dict
import inspect
import yaml
import os

from ..save_mixin.basemixin import BaseSaveMixin


def load_components(manifest):
    components = []
    for comp in manifest:
        
        module = importlib.import_module(f"ANIDSC.{comp['type']}")
        component_cls = getattr(module, comp["class"])
        if comp.get("file", False):
            file_path = comp["file"]
            comp = component_cls.load(file_path)
        else:
            comp = component_cls(**comp.get("attrs", {}))
        components.append( comp)
    return components


class YamlSaveMixin(BaseSaveMixin):
    @property
    def save_type(self):
        return "yaml"

    def save(self):
        # save individual components
        for component in self.components:
            component.save()

        # save manifest
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, "w") as out_file:
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
            manifest = input_data
        else:
            raise TypeError("Unknown input_data format")

        return cls(load_components(**manifest["attrs"]))
