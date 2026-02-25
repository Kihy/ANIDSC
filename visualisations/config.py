"""Global constants and configuration."""
from pathlib import Path
from typing import Dict

ROOT = Path("/workspace/intrusion_detection/runs")

MAC_TO_DEVICE: Dict[str, str] = {
    "52:8e:44:e1:da:9a": "Cam",
    "be:7b:f6:f2:1b:5f": "Attacker",
    "5e:ea:b2:63:fc:aa": "Google-Nest",
    "52:a8:55:3e:34:46": "Lenovo_Bulb",
    "42:7f:83:17:55:c0": "Raspberry Pi",
    "a2:bd:fa:b5:89:92": "Smart_Clock",
    "22:c9:ca:f6:da:60": "Smartphone_1",
    "d2:19:d4:e9:94:86": "Smartphone_2",
    "7e:d1:9d:c4:d1:73": "SmartTV",
}

LAYER_COLORS = {
    "Physical": "#FF8C42",
    "Internet": "#4ECDC4",
    "Transport": "#95E06C",
}
