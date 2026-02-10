"""Converter registry and conversion logic"""
from typing import Any, Type, Dict, Optional
from .base import TypeAdapter

class ConverterRegistry:
    """
    Central registry for type adapters.
    Adapters are automatically registered via AutoRegisterAdapter.__init_subclass__
    """
    
    _adapters: Dict[type, Type[TypeAdapter]] = {}
    
    @classmethod
    def register(cls, adapter: Type[TypeAdapter]) -> None:
        """
        Register a type adapter.
        
        Args:
            adapter: Adapter class to register
        """
        try:
            target = adapter.target_type()
            cls._adapters[target] = adapter
        except Exception as e:
            # Silently fail if target_type() raises (e.g., missing dependencies)
            pass
    
    @classmethod
    def unregister(cls, target_type: type) -> None:
        """
        Unregister an adapter.
        
        Args:
            target_type: Type to unregister adapter for
        """
        cls._adapters.pop(target_type, None)
    
    @classmethod
    def get_adapter(cls, target_type: type) -> Optional[Type[TypeAdapter]]:
        """
        Get adapter for a target type.
        
        Args:
            target_type: Type to get adapter for
            
        Returns:
            Adapter class or None if not found
        """
        return cls._adapters.get(target_type)
    
    @classmethod
    def has_adapter(cls, target_type: type) -> bool:
        """
        Check if an adapter is registered for a type.
        
        Args:
            target_type: Type to check
            
        Returns:
            True if adapter exists
        """
        return target_type in cls._adapters
    
    @classmethod
    def convert(cls, value: Any, target_type: type) -> Any:
        """
        Convert value to target_type using registered adapter.
        
        Args:
            value: Value to convert
            target_type: Type to convert to
            
        Returns:
            Converted value
            
        Raises:
            TypeError: If no adapter is registered or conversion fails
        """
        # Check if already correct type
        if isinstance(value, target_type):
            return value
        
        # Get adapter
        adapter = cls.get_adapter(target_type)
        if adapter is None:
            raise TypeError(
                f"No adapter registered for type {target_type.__name__}. "
                f"Available adapters: {[t.__name__ for t in cls._adapters.keys()]}"
            )
        
        # Check if adapter can convert
        if not adapter.can_convert(value):
            raise TypeError(
                f"Adapter {adapter.__name__} cannot convert "
                f"{type(value).__name__} to {target_type.__name__}"
            )
        
        # Perform conversion
        try:
            return adapter.convert(value)
        except Exception as e:
            raise TypeError(
                f"Conversion failed: {type(value).__name__} → {target_type.__name__}: {e}"
            ) from e
    
    @classmethod
    def list_adapters(cls) -> Dict[str, str]:
        """
        List all registered adapters.
        
        Returns:
            Dict mapping target type names to adapter class names
        """
        return {
            target_type.__name__: adapter.__name__
            for target_type, adapter in cls._adapters.items()
        }
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters (mainly for testing)"""
        cls._adapters.clear()