"""Auto-cast decorator for automatic type conversion"""
from functools import wraps
from typing import get_type_hints, Callable, Any, TypeVar, get_origin
import inspect
from .registry import ConverterRegistry

T = TypeVar('T')


def auto_cast(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that automatically converts function arguments to their annotated types.
    
    Uses the ConverterRegistry to find appropriate adapters for each type hint.
    If no adapter is available or conversion fails, passes the argument unchanged.
    
    Example:
        @auto_cast
        def process(data: torch.Tensor, mask: np.ndarray) -> tuple:
            return data, mask
        
        # Now you can call with any convertible type:
        process([[1, 2]], [True, False])  # lists auto-converted to tensor and array
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with auto-conversion
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints for the function
        try:
            hints = get_type_hints(func)
        except Exception:
            # If we can't get hints, just call the function normally
            return func(*args, **kwargs)
        
        # Get parameter names from function signature
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        # Convert positional arguments
        new_args = []
        for i, arg in enumerate(args):
            if i >= len(param_names):
                # More args than parameters, just pass through
                new_args.append(arg)
                continue
            
            param_name = param_names[i]
            
            if param_name not in hints:
                raise ValueError("No type specified")
            
            # Skip if it's the return type
            if param_name == 'return':
                new_args.append(arg)
                continue
            
            hint = hints[param_name]
            
            # Skip generic types (like List, Dict, etc.) - only handle concrete types
            if get_origin(hint) is not None:
                new_args.append(arg)
                continue
            
            # Try to convert using registry
            try:
                converted = ConverterRegistry.convert(arg, hint)
                new_args.append(converted)
            except TypeError:
                # No adapter registered or conversion failed, pass through unchanged
                new_args.append(arg)
        
        # Convert keyword arguments
        new_kwargs = {}
        for key, value in kwargs.items():
            if key not in hints or key == 'return':
                new_kwargs[key] = value
                continue
            
            hint = hints[key]
            
            # Skip generic types
            if get_origin(hint) is not None:
                new_kwargs[key] = value
                continue
            
            try:
                converted = ConverterRegistry.convert(value, hint)
                new_kwargs[key] = converted
            except TypeError:
                new_kwargs[key] = value
        
        return func(*new_args, **new_kwargs)
    
    return wrapper


def auto_cast_method(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for class methods (handles 'self' or 'cls' parameter).
    
    Similar to auto_cast but skips the first parameter which is assumed to be
    'self' (for instance methods) or 'cls' (for class methods).
    
    Example:
        class Processor:
            @auto_cast_method
            def process(self, data: torch.Tensor) -> torch.Tensor:
                return data * 2
        
        processor = Processor()
        result = processor.process([[1, 2], [3, 4]])  # list auto-converted to tensor
    
    Args:
        func: Method to decorate
        
    Returns:
        Wrapped method with auto-conversion
    """
    @wraps(func)
    def wrapper(self_or_cls, *args, **kwargs):
        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            return func(self_or_cls, *args, **kwargs)
        
        # Get parameter names, excluding first (self/cls)
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())[1:]  # Skip first parameter
        
        # Convert positional arguments
        new_args = []
        for i, arg in enumerate(args):
            if i >= len(param_names):
                new_args.append(arg)
                continue
            
            param_name = param_names[i]
            
            if param_name not in hints:
                raise ValueError("No type specified")
            
            if param_name == 'return':
                new_args.append(arg)
                continue
            
            hint = hints[param_name]
            
            # Skip generic types
            if get_origin(hint) is not None:
                new_args.append(arg)
                continue
            
            
            converted = ConverterRegistry.convert(arg, hint)
            new_args.append(converted)
            
        
        # Convert keyword arguments
        new_kwargs = {}
        for key, value in kwargs.items():
            if key not in hints or key == 'return':
                new_kwargs[key] = value
                continue
            
            hint = hints[key]
            
            if get_origin(hint) is not None:
                new_kwargs[key] = value
                continue
            
            try:
                converted = ConverterRegistry.convert(value, hint)
                new_kwargs[key] = converted
            except TypeError:
                new_kwargs[key] = value
        
        return func(self_or_cls, *new_args, **new_kwargs)
    
    return wrapper


def selective_auto_cast(**conversions: Callable[[Any], Any]):
    """
    Decorator that only converts specific parameters using custom conversion functions.
    
    This is useful when you want fine-grained control over which parameters get converted
    and how they get converted.
    
    Example:
        @selective_auto_cast(
            timestamp=lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x,
            tags=lambda x: x.split(',') if isinstance(x, str) else x
        )
        def process(timestamp: datetime, tags: list, score: float):
            return timestamp, tags, score
        
        process("2024-01-15T10:30:00", "python,ai,ml", 95.5)
    
    Args:
        **conversions: Keyword arguments mapping parameter names to converter functions
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Convert positional arguments
            new_args = []
            for i, arg in enumerate(args):
                if i >= len(param_names):
                    new_args.append(arg)
                    continue
                
                param_name = param_names[i]
                
                if param_name in conversions:
                    try:
                        new_args.append(conversions[param_name](arg))
                    except Exception:
                        new_args.append(arg)
                else:
                    new_args.append(arg)
            
            # Convert keyword arguments
            new_kwargs = {}
            for key, value in kwargs.items():
                if key in conversions:
                    try:
                        new_kwargs[key] = conversions[key](value)
                    except Exception:
                        new_kwargs[key] = value
                else:
                    new_kwargs[key] = value
            
            return func(*new_args, **new_kwargs)
        
        return wrapper
    return decorator