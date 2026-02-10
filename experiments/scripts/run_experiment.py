import argparse
import json
from itertools import product
from typing import Any, Dict, List


from ANIDSC.utils.helper import generate_cartesian_configs
from ANIDSC.utils.run_script import run_file
from ANIDSC.utils.dataset_registry import dataset_registry




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run feature extraction and pipeline detection"
    )

    parser.add_argument(
        "dataset",type=str, 
        help="name of dataset (e.g., test)",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Pipeline Dictionary as JSON string"
    )

    args = parser.parse_args()
    config = json.loads(args.config)

    
    for cfg in generate_cartesian_configs(config):
        run_file(dataset_registry.get(args.dataset), cfg)
