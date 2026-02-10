from itertools import product
from typing import Generator, Tuple, Callable, Dict

class DatasetRegistry:
    """Registry for dataset generators."""
    
    def __init__(self):
        self._datasets: Dict[str, Callable] = {}
    
    def register(self, name: str):
        """Decorator to register a dataset generator."""
        def decorator(func: Callable) -> Callable:
            self._datasets[name] = func
            return func
        return decorator
    
    def get(self, name: str) -> Callable:
        """Get a dataset generator by name."""
        if name not in self._datasets:
            available = ", ".join(sorted(self._datasets.keys()))
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        return self._datasets[name]
    
    def list_datasets(self) -> list[str]:
        """List all registered dataset names."""
        return sorted(self._datasets.keys())


# Create global registry instance
dataset_registry = DatasetRegistry()


# Register datasets using decorator
@dataset_registry.register("test_dataset")
def test_dataset() -> Generator[Tuple[str, str, str], None, None]:
    """Test dataset with benign and attack files."""
    # benign files
    yield "new", "benign_lenovo_bulb", "../datasets/test_data"

    # Attack files (subsequent runs)
    for attack in [
        "malicious_ACK_Flooding",
        "malicious_Port_Scanning",
        "malicious_Service_Detection",
    ]:
        yield "loaded", attack, "../datasets/test_data"
        
@dataset_registry.register("uq_dataset")
def uq_dataset():
    
    DEVICES = [
        "Cam_1",
        "Google-Nest-Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        "Smart_Clock_1",
        "Smartphone_1",
        "Smartphone_2",
        "SmartTV",
    ]
    ATTACKS = [
        "ACK_Flooding",
        "ARP_Spoofing",
        "Port_Scanning",
        "Service_Detection",
        "SYN_Flooding",
        "UDP_Flooding",
    ]
   
    yield "new", "benign_samples/whole_week", "../datasets/UQ-IoT-IDS"

    # Attack files (subsequent runs)
    for device, attack in product(DEVICES, ATTACKS):

        yield "loaded", f"attack_samples/{device}/{attack}" , "../datasets/UQ-IoT-IDS"