"""Plot registry.

To register a new plot type:
  1. Create plots/my_plot.py with a class that subclasses BasePlot and ends with 'Plot'.
  2. That's it! The class will be automatically discovered and registered.

The rest of the dashboard picks up changes automatically.
"""

import importlib
import inspect
from pathlib import Path
from plots.base import BasePlot

# Automatically discover all plot classes ending with 'Plot'
_current_dir = Path(__file__).parent
_registry = []

for py_file in sorted(_current_dir.glob("*.py")):
    if py_file.name.startswith("_") or py_file.stem == "base":
        continue
    
    module_name = f"plots.{py_file.stem}"
    try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (name.endswith("Plot") and 
                name != "BasePlot" and 
                issubclass(obj, BasePlot) and 
                obj is not BasePlot):
                _registry.append(obj)
    except Exception:
        continue

# Ordered list — defines the button order in the sidebar.
# Sorted alphabetically by class name
REGISTRY = sorted(_registry, key=lambda cls: cls.__name__)
