from collections import defaultdict
import copy
from datetime import datetime
import gc
import os
from pathlib import Path
from ANIDSC.utils.helper import load_yaml

import optuna

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch_geometric as pyg
from ..pipeline import Pipeline
import yaml
from ..templates import get_template 
from .logger import setup_logging
from .helper import print_dictionary

def _to_yaml_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_yaml_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def setup_experiment(dataset, config, run_id=None, prev_pipeline=None):
    """
    Common setup for experiments: set run identifier, setup logging, and return job_id.
    
    Args:
        dataset: Dataset object with location attribute
        config: Pipeline configuration dictionary
        run_id: Optional run identifier
        prev_pipeline: Optional previous pipeline name to be used as input
    Returns:
        Tuple of (job_id, config) with run_identifier set
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    
    # Set run identifier priority: run_id in command > run_id in config > SLURM_JOB_ID > timestamp
    if run_id is None:
        run_id = config.get("run_identifier") 
    if run_id is None: 
        run_id = job_id 
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    config["run_identifier"] = run_id
    
    if prev_pipeline:
        config["prev_pipeline"] = prev_pipeline
    
    # Allow pipeline_name to be overridden by pipeline_vars, otherwise use template_name
    pipeline_name=config.get("pipeline_name", config["template_name"].removesuffix("-template"))
    
    config["pipeline_name"] = pipeline_name
    
    # Setup logging
    log_dir = f"runs/{dataset.location}/{run_id}/{config['pipeline_name']}/logs.out"
    setup_logging(log_dir)
    
    return config


def yaml_align(data, indent=0):
    lines = []

    if isinstance(data, dict):
        # Longest key at this level
        max_len = max(len(str(k)) for k in data.keys())

        for k, v in data.items():
            key = str(k).ljust(max_len)
            prefix = " " * indent

            if isinstance(v, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_align(v, max_len + indent + 2))
            else:
                lines.append(f"{prefix}{key}: {v}")

    else:
        lines.append(" " * indent + str(data))

    return lines


def pprint(data):
    """Return fully aligned YAML-style string for the entire structure."""
    print("\n".join(yaml_align(data, indent=0)))
    



def _load_dataframes(results):
    """Load, validate, and return cleaned DataFrames from result CSV paths."""

    combined_dfs = []
    for result in results:
        csv_path = result["result_path"]

        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            print(f"Skipping {csv_path}: file does not exist or is empty")
            continue

        df = pd.read_csv(csv_path).replace(np.inf, np.finfo(np.float64).max).dropna()

        # add file_name to df for later grouping
        df["file_name"] = result["file_name"] 
        combined_dfs.append(df)
    
    return pd.concat(combined_dfs, ignore_index=True)    
        


def compute_column_summary(combined):
    """Aggregate columns across all valid CSVs according to _AGG_FUNCS."""
        
    _AGG_FUNCS = {
        "process_time": "mean",
        "detection_rate": "mean",
        "median_score": "mean",
        "median_threshold": "mean",
        "pos_count": "sum",
        "batch_size": "sum",
    }
    
    column_summary={
        col: getattr(combined[col], agg)().item()
        for col, agg in _AGG_FUNCS.items()
        if col in combined.columns
    }
    
    return column_summary



def generate_summary(results):
    
    df=_load_dataframes(results)
    
    col_summary=compute_column_summary(df) 
    
    ap_summary, file_acc = compute_detection_metrics(df)
    

    return {
        "run_summary":{
            **col_summary,
            **ap_summary
            },
        "file_accuracy": file_acc
        }
        

def compute_detection_metrics(df, benign_prefix=("benign",), attack_prefix=("attack", "malicious")):
    """
    Compute detection metrics (AP and accuracy) from a DataFrame with file_name, score, and detection_rate columns.
    
    AP metrics:
    - pooled_ap: all negatives merged into one ranked list
    - macro_ap: average of per-pair APs
    - weighted_ap: weighted average by number of samples
    
    File accuracy:
    - For benign files: accuracy = 1 - detection_rate (lower false positives)
    - For malicious files: accuracy = detection_rate (higher detections)
    
    Returns dict with AP metrics and file_accuracy DataFrame
    """
    benign_mask = df["file_name"].str.startswith(benign_prefix)
    attack_mask = df["file_name"].str.startswith(attack_prefix)

    if not benign_mask.any():
        raise ValueError("No benign files found.")
    if not attack_mask.any():
        raise ValueError("No attack files found.")

    benign_scores = df.loc[benign_mask, "median_score"].to_numpy()

    attack_scores_list = [
        g.to_numpy()
        for _, g in df.loc[attack_mask].groupby("file_name")["median_score"]
    ]

    def ap_vs_benign(attack_scores):
        scores = np.concatenate([benign_scores, attack_scores])
        labels = np.repeat([1, 0], [len(benign_scores), len(attack_scores)])
        return average_precision_score(labels, scores)

    pair_aps = [ap_vs_benign(s) for s in attack_scores_list]

    ap_results = {
        "pooled_ap": ap_vs_benign(np.concatenate(attack_scores_list)),
        "macro_ap": np.mean(pair_aps),
        "weighted_ap": np.average(pair_aps, weights=[len(s) for s in attack_scores_list]),
    }
    
    file_acc={}
    # group by file_name and compute accuracy
    for file_name, group in df.groupby("file_name"):
        detection_rate=group["pos_count"].sum()/group["batch_size"].sum()
        if file_name.startswith(benign_prefix):
            file_acc[file_name] = 1 - detection_rate
        elif file_name.startswith(attack_prefix):
            file_acc[file_name] = detection_rate
        else:
            raise ValueError(f"Unknown file type for {file_name}")
    
    return ap_results, file_acc



def run(file_iterator, pipeline_vars, return_pipeline=False):
    pipelines = []
    results=[]
    
    pipeline_vars["dataset_name"] = file_iterator.location
        
    template_name=pipeline_vars["template_name"]
    pipeline_name=pipeline_vars.get("pipeline_name", template_name.removesuffix("-template"))
    pipeline_vars["pipeline_name"] = pipeline_name
    
    if "model_params" in pipeline_vars:
        model_params = pipeline_vars["model_params"]
        
        # If it's a string, load it with load_yaml
        if isinstance(model_params, str):
            model_params = load_yaml(model_params)
        
        if isinstance(model_params, dict) and any(isinstance(v, dict) for v in model_params.values()):
            pipeline_vars["model_params"] = {k: v["default"] for k, v in model_params.items()}
    
    # Print configuration
    print("Running experiment with the following configuration:")
    print_dictionary(pipeline_vars)    
    print(f"Start time: {datetime.now()}")
    print("\n" + "=" * 50 + "\n")
    
    for state, file in file_iterator:        
        pipeline_vars["file_name"] = file
        
        print(f"Running {state} {template_name} on {file}:")
        print("-"*50)

        if state == "new":
            pipeline = Pipeline.load(get_template(**pipeline_vars))
            pipeline.setup()

            benign_path = pipeline.save_path

        elif state == "loaded":

            with open(benign_path) as f:
                manifest = yaml.safe_load(f)

            # datasource is always 0
            manifest["attrs"]["manifest"][0]["attrs"]["file_name"] = pipeline_vars[
                "file_name"
            ]
            manifest["attrs"]["manifest"][0]["file"] = None
            manifest["attrs"]["run_identifier"] = pipeline_vars["run_identifier"]

            pipeline = Pipeline.load(manifest)
            pipeline.setup()
        else:
            raise ValueError("Unknown State", state)
        pipeline.start()
        results.append(pipeline.information_dict)

        print("\n" + "=" * 50 + "\n")
        print(f"Execution completed successfully at {datetime.now()}")
        
        if return_pipeline:
            pipelines.append(pipeline)
        else:
            del pipeline 
            gc.collect()


    if pipeline_vars.get("gen_summary"):
        # save summary as yaml
        summary=generate_summary(results)
        summary["dataset_name"] = pipeline_vars["dataset_name"]
        summary["pipeline_name"] = pipeline_vars["pipeline_name"]
        summary["run_identifier"] = pipeline_vars["run_identifier"]
        summary = _to_yaml_safe(summary)
        

        summary_path = Path(f"runs/{file_iterator.location}/{pipeline_vars['run_identifier']}/{pipeline_vars['pipeline_name']}/summary.yaml")
        summary_path.parent.mkdir(parents=True, exist_ok=True) 
        with open(summary_path, "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
    else:
        summary=None
    
    
    return pipelines, summary

def sample_from_spec(trial, model_spec):
    params = {}
    model_spec=load_yaml(model_spec) 

    for name, spec in model_spec.items():
        
        t = spec["type"]

        if t == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )

        elif t == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"]
            )

        elif t == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                spec["choices"]
            )

        else:
            raise ValueError(f"Unsupported type {t}")

    return params

def make_objective(pipeline_vars, file_iterator):
    def objective(trial):
        # Set random seed for reproducibility
        pyg.seed.seed_everything(trial.number)
        
        # Avoid cross-trial contamination
        trial_vars = copy.deepcopy(pipeline_vars)

        # Sample hyperparameters
        trial_vars["model_params"] = sample_from_spec(
            trial, pipeline_vars["model_params"]
        )


        trial_vars["run_identifier"] = f"{trial_vars['run_identifier']}/trial_{trial.number}"

        _, summary = run(file_iterator, trial_vars)

        return summary["run_summary"]["pooled_ap"]+summary["run_summary"]["macro_ap"]+summary["run_summary"]["weighted_ap"]

    return objective



def tune(file_iterator, pipeline_vars, n_trials):
    print(f"Tuning models with {n_trials} trials...")
    objective = make_objective(pipeline_vars, file_iterator)
    
    db_path = Path(f"runs/{file_iterator.location}/optuna.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(study_name=f"{pipeline_vars['run_identifier']}_{pipeline_vars['pipeline_name']}",
        storage=f"sqlite:///{db_path}",   # persistent storage
        load_if_exists=True,
        sampler=sampler,
        direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study