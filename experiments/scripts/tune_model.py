import argparse
from datetime import datetime
import json
import os



from ANIDSC.utils.helper import generate_cartesian_configs, load_json
from ANIDSC.utils.logger import setup_logging
from ANIDSC.utils.run_script import tune_hyperparameters
from ANIDSC.utils.dataset_registry import dataset_registry




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for a pipeline"
    )

    parser.add_argument(
        "dataset",type=str, 
        help="name of dataset (e.g., test_dataset)",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Pipeline Dictionary as JSON string"
    )
    
    parser.add_argument(
        "--hyperparam_spec", type=str, required=True, help="Hyperparameter specification as JSON string"
    )
    parser.add_argument(
        "--run_id", type=str, required=False, help="Run identifier for tracking experiments"
    )


    args = parser.parse_args()
    
    #get dataset
    dataset = dataset_registry.get(args.dataset)
    
    # load json
    config = load_json(args.config)
    hyperparam_spec=load_json(args.hyperparam_spec)

    job_id = os.environ.get("SLURM_JOB_ID")
    
    # if no run identifier in config, use job id
    if config.get("run_identifier") is None:
        if args.run_id is not None:
            config["run_identifier"] = args.run_id
        elif job_id is not None:
            config["run_identifier"] = job_id
        else:
            config["run_identifier"] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # set up logging
    log_dir=f"runs/{dataset.location}/{config['pipeline_name']}/{config['run_identifier']}/logs.out"
    setup_logging(log_dir)


    for cfg in generate_cartesian_configs(config):
        print("Running experiment with the following configuration:")
        print(f"Job ID: {job_id}")
        print(f"Config content: {cfg}")
        print(f"Log directory: {log_dir}")
        print(f"Hyperparam content: {hyperparam_spec}")
        print(f"Start time: {datetime.now()}")

        print("\n ========================================================== \n")
        

        study=tune_hyperparameters(dataset, cfg, hyperparam_spec)
        
        print("\n ========================================================== \n")

        print(f"End time: {datetime.now()}")