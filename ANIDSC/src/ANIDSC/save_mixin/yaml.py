import importlib
from pathlib import Path
from typing import Any, Dict
import inspect
import yaml
import os

from ..save_mixin.basemixin import BaseSaveMixin

from ..utils.helper import load_yaml



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
        spec=load_yaml(input_data)

        return cls(**spec['attrs'])
