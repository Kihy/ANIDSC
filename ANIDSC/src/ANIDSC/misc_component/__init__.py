import os
import importlib
import inspect

# Initialize __all__ to define the public API
__all__ = []

# Get the directory of the current file
package_dir = os.path.dirname(__file__)

# Iterate over all Python files in the directory
for filename in os.listdir(package_dir):
    # Skip __init__.py and non-Python files
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove .py extension
        # Import the module
        module = importlib.import_module(f'.{module_name}', package=__name__)
        # Iterate over all attributes of the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # Check if the attribute is a class defined in this module
            if inspect.isclass(attr) and attr.__module__ == module.__name__:
                # Add the class to the package namespace
                globals()[attr_name] = attr
                __all__.append(attr_name)