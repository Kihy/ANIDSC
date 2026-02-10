"""
Import all adapters - this triggers auto-registration via __init_subclass__
"""

# Explicitly import each adapter module to trigger class definition
import importlib
import pkgutil
from pathlib import Path

# Get all .py files in this directory
_adapter_modules = []
_adapters_dir = Path(__file__).parent

for file in _adapters_dir.glob('*_adapter.py'):
    module_name = file.stem
    try:
        # Import the module - this triggers class definition and auto-registration
        mod = importlib.import_module(f'.{module_name}', package=__package__)
        _adapter_modules.append(mod)
    except ImportError as e:
        # Dependency not available (e.g., torch not installed)
        pass

# Try to expose the adapter classes if they exist
__all__ = []


from .tensor_adapter import TensorAdapter
__all__.append('TensorAdapter')



from .numpy_adapter import NumpyAdapter
__all__.append('NumpyAdapter')

from .dictlist_adapter import RecordListAdapter
__all__.append('RecordListAdapter')

