import importlib
from pathlib import Path
from typing import Any, Dict
import inspect
import yaml
import os


class YamlSaveMixin:
    def save(self):
        manifest = {"components": {}}

        # save individual components
        for key, component in self.components.items():
            save_path = component.save()

            comp_dict = {
                "class": component.component_name,
                "attrs": component.get_save_attr(),
            }
            if save_path:
                comp_dict["file"] = save_path

            # add components to manifest
            manifest["components"][key]=comp_dict

        # save manifest
        manifest_path = Path(self.get_save_path("yaml"))

        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, "w") as out_file:
            yaml.safe_dump(manifest, out_file, indent=4, sort_keys=False)

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
        
        components = {}
        for type, meta in manifest["components"].items():
            module = importlib.import_module(f"ANIDSC.{type}")
            component_cls = getattr(module, meta["class"])
            if meta.get("file", False):
                file_path = meta["file"]
                comp = component_cls.load(file_path)
            else:
                comp = component_cls(**meta.get("attrs", {}))
            components[type] = comp
        return cls(components)
