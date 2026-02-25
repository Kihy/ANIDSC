from collections import defaultdict
import copy
import gc
import os
from pathlib import Path


import optuna

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch_geometric as pyg
from ..pipeline import Pipeline
import yaml
from ..templates import get_template 

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

def generate_summary(results):
    scores_dict = defaultdict(dict)
    
    for result in results:
        csv_path = result["result_path"]

        if os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path)
        else:
            print(f"Skipping {csv_path}: file is empty")
        if len(df) < 5:
            continue

        # filter
        df = df.replace(np.inf, np.finfo(np.float64).max).dropna()

        if len(df) == 0:
            print(f"skipping {csv_path} as it contains no finite value")
        else:
            scores_dict[result["file_name"]]= df["median_score"].values
    ap_results = compute_ap(scores_dict)
    return ap_results


def compute_ap(
    scores_dict: dict,
    benign_prefix: tuple = ("benign",),
    attack_prefix: tuple = ("attack", "malicious"),
) -> pd.DataFrame:
    """
    Compute Pooled, Macro, and Weighted Average Precision for each model.

    Args:
        scores_dict:   Nested dict of {model_name: {file_name: np.array of scores}}.
        benign_prefix: Filename prefixes identifying benign (positive) files.
        attack_prefix: Filename prefixes identifying attack (negative) files.

    Returns:
        DataFrame with columns [model_name, pooled_ap, macro_ap, weighted_ap].
    """

    def _make_labels_scores(benign_scores, malicious_scores):
        """Concatenate scores and infer labels: positive=1, negative=0."""
        all_scores = [benign_scores, malicious_scores]
        scores = np.concatenate(all_scores)
        labels = np.concatenate([
            np.full(len(s), i, dtype=int) for i, s in enumerate(all_scores)
        ])
        return scores, labels

    
    benign_files = [f for f in scores_dict if f.startswith(benign_prefix)]
    attack_files = [f for f in scores_dict if f.startswith(attack_prefix)]

    if not benign_files:
        raise ValueError(f"No benign files found.")
    if not attack_files:
        raise ValueError(f"No attack files found.")

    # Aggregate all positive scores across benign files
    benign_scores = np.concatenate([scores_dict[f] for f in benign_files])

    # --- Pooled AP: all negatives merged into one ranked list ---
    all_malicious_scores = np.concatenate([scores_dict[f] for f in attack_files])
    pooled_scores, pooled_labels = _make_labels_scores(benign_scores, all_malicious_scores)
    pooled_ap = average_precision_score(pooled_labels, pooled_scores)

    # --- Macro AP & Weighted AP: per-pair, then aggregate ---
    pair_aps, pair_weights = [], []
    for attack_file in attack_files:
        malicious_scores = scores_dict[attack_file]
        scores, labels = _make_labels_scores(benign_scores, malicious_scores)
        pair_aps.append(average_precision_score(labels, scores))
        pair_weights.append(len(malicious_scores))

    macro_ap = np.mean(pair_aps)
    weighted_ap = np.average(pair_aps, weights=pair_weights)


    return {
        "pooled_ap": pooled_ap,
        "macro_ap": macro_ap,
        "weighted_ap": weighted_ap,
    }

def run_file(file_iterator, pipeline_vars, return_pipeline=False):
    pipelines = []
    results=[]
    for state, file in file_iterator:        
        pipeline_vars["file_name"] = file
        pipeline_vars["dataset_name"] = file_iterator.location
        pipeline_name=pipeline_vars["pipeline_name"]
                
        print(f"Running {state} pipeline {pipeline_name}:")
        print("-"*50)
        pprint(pipeline_vars)
        print("-"*50)

        if state == "new":
            pipeline = Pipeline.load(get_template(pipeline_name, **pipeline_vars))
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

        print("Execution completed successfully!")
        
        if return_pipeline:
            pipelines.append(pipeline)
        else:
            del pipeline 
            gc.collect()

    # save summary as dataframe
    summary=generate_summary(results)
    summary["dataset_name"] = pipeline_vars["dataset_name"]
    summary["pipeline_name"] = pipeline_vars["pipeline_name"]
    summary["run_identifier"] = pipeline_vars["run_identifier"]
    
    summary = pd.DataFrame([summary])
    summary_path = Path(f"runs/{file_iterator.location}/{pipeline_vars['pipeline_name']}/{pipeline_vars['run_identifier']}/summary.csv")
    summary.to_csv(summary_path, index=False)
    
    
    return pipelines, summary

def sample_from_spec(trial, hyperparam_spec):
    params = {}

    for name, spec in hyperparam_spec.items():
        if name == "n_trials":
            continue  # Skip n_trials as it's not a hyperparameter to sample
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

def make_objective(pipeline_vars, file_iterator, hyperparam_spec):
    def objective(trial):
        # Set random seed for reproducibility
        pyg.seed.seed_everything(trial.number)
        
        # Avoid cross-trial contamination
        trial_vars = copy.deepcopy(pipeline_vars)

        # Sample hyperparameters
        trial_vars["model_params"] = sample_from_spec(
            trial, hyperparam_spec
        )

        
        trial_vars["run_identifier"] = f"{trial_vars['run_identifier']}/trial_{trial.number}"

        _, summary = run_file(file_iterator, trial_vars)

        return summary["pooled_ap"]+summary["macro_ap"]+summary["weighted_ap"]

    return objective



def tune_hyperparameters( file_iterator, pipeline_vars, hyperparam_spec):
    objective = make_objective(pipeline_vars, file_iterator, hyperparam_spec)
    
    db_path = Path(f"runs/{file_iterator.location}/optuna.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(study_name=f"{pipeline_vars['pipeline_name']}_{pipeline_vars['run_identifier']}",
        storage=f"sqlite:///{db_path}",   # persistent storage
        load_if_exists=True,
        sampler=sampler,
        direction="maximize")
    study.optimize(objective, n_trials=hyperparam_spec["n_trials"])
    return study