#!/usr/bin/env python3
"""
Unified CLI for running and tuning ML pipelines.

Usage:
    python3 main.py run <dataset> --config <config.yaml> [--run_id <id>]
    python3 main.py tune <dataset> --config <config.yaml> --hyperparam_spec <spec.yaml> [--run_id <id>]
"""

import argparse
from datetime import datetime

from ANIDSC.utils.helper import load_yaml
from ANIDSC.utils import run_script
from ANIDSC.utils.dataset_registry import dataset_registry




def execute_command(args, kwargs):
    """Generic command executor that dispatches to run_script functions."""
    dataset = dataset_registry.get(args.dataset)

    # Setup experiment and logging, may update run_id in config
    config = run_script.setup_experiment(dataset, args.config, args.run_identifier, args.prev_pipeline)
    
    # Get the command function from run_script by name
    command_func = getattr(run_script, args.command)
    
    # Execute command
    command_func(dataset, config, **kwargs)
    



def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for running and tuning ML pipelines"
    )
    
    parser.add_argument("command", type=str, help="Operation to perform (run or tune)")
    parser.add_argument("dataset", type=str, help="Name of dataset (e.g., test_dataset)")
    parser.add_argument("--prev_pipeline", type=str, required=False, help="Previous pipeline to use as input (e.g., frequency-extraction/box_plot_test)")
    
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file or dictionary")
    parser.add_argument("--run_identifier", type=str, required=False, help="Run identifier for tracking experiments")
    
    
    args, unknown = parser.parse_known_args()

    #load config from yaml file or parse as dictionary
    args.config=load_yaml(args.config)
    
    # Convert unknown args into kwargs
    kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-")
        value = unknown[i + 1]
        kwargs[key] = load_yaml(value)

    print(kwargs)
    execute_command(args, kwargs)


if __name__ == "__main__":
    main()
