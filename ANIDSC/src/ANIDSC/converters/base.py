"""Base protocol and auto-registration"""
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class TypeAdapter(Protocol):
    """Protocol that all type adapters must implement"""
    
    @staticmethod
    def can_convert(value: Any) -> bool: ...
    
    @staticmethod
    def convert(value: Any) -> Any: ...
    
    @staticmethod
    def target_type() -> type: ...


class AutoRegisterAdapter:
    """
    Base class that auto-registers adapters when they're defined.
    Just inherit from this class and the adapter will be automatically registered!
    """
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Import here to avoid circular dependency
        from .registry import ConverterRegistry
        
        # Only register if it's a concrete implementation (has all required methods)
        if all(hasattr(cls, method) for method in ['can_convert', 'convert', 'target_type']):
            try:
                ConverterRegistry.register(cls)
            except Exception:
                # Silently fail if dependencies not available
                pass