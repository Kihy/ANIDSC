from datetime import datetime
from pathlib import Path
from ANIDSC.pipeline import Pipeline
from ANIDSC.utils.run_script import run, tune
from behave import given, when, then
from shutil import rmtree
import fsspec
import pandas as pd
import os
import yaml
from ANIDSC.utils.dataset_registry import dataset_registry

@given("Pipeline variable: {var_name} -> {var_value}")
def step_given_pipeline_var(context, var_name, var_value):
    context.pipeline_vars[var_name] = _parse_value(var_value)
    

    
@given("Run Identifier: {run_identifier}")
def step_given_run_identifier(context, run_identifier):
    
    context.pipeline_vars["run_identifier"] = run_identifier

def _parse_value(value: str):
    """Try to parse as YAML, fall back to raw string."""
    try:
        return yaml.safe_load(value)
    except (yaml.YAMLError, TypeError):
        return value

@given("Hyperparameter Tuning variable: {var_name} -> {var_value}")
def step_given_hyperparameter_tuning_var(context, var_name, var_value):
    context.tuning_vars[var_name] = _parse_value(var_value)



@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the results are written")
def step_then_results_saved(context):
    for pipeline in context.pipelines:

        p_name = pipeline.name.split("->")
        for i in p_name:
            if i.endswith("ResultWriter"):
                results_path = str(pipeline.get_attr_by_name(i, "feature_path"))
                if results_path.endswith("csv.zst"):
                    with fsspec.open(results_path, "rt", compression="zstd") as f:
                        csv_df = pd.read_csv(f)
                        assert len(csv_df) > 0, f"{results_path} is empty"
                elif results_path.endswith("ndjson.zst"):
                    assert os.stat(results_path).st_size > 0, f"{results_path} is empty"


@then("the components are saved")
def step_then_components_are_saved(context):

    for pipeline in context.pipelines:
        manifest_path = str(pipeline.save_path)

        loaded_pipeline = Pipeline.load(manifest_path)

        # do not call setup because attributes in setup are assumed to be created dynamically

        assert loaded_pipeline == pipeline


@given("The test_data file iterator")
def step_given_dataset_and_file(context):
    context.file_iterator = dataset_registry.get("test_dataset")
    context.pipeline_vars = {}
    context.tuning_vars = {}


@then("the optuna database is created")
def step_then_optuna_db_created(context):
    dataset=context.file_iterator.location
    pipeline_name=context.pipeline_vars["pipeline_name"]
    run_id=context.pipeline_vars["run_identifier"]
    db_path = f"runs/{dataset}/optuna.db"
    assert os.path.exists(db_path), f"Optuna database {db_path} does not exist"

@given("a {pipeline_name} pipeline")
def step_given_pipeline(context, pipeline_name):

    context.pipeline_vars["pipeline_name"] = pipeline_name


@when("the pipeline starts")
def step_when_pipeline_starts(context):
    context.pipelines, summary = run(
        context.file_iterator, context.pipeline_vars, return_pipeline=True
    )

@when("the pipeline starts with optuna tuning enabled with 20 trials")
def step_when_pipeline_starts_with_optuna(context):
    study = tune(
        context.file_iterator, context.pipeline_vars, n_trials=20
    )

@given("folders in {dataset} with run identifier are empty")
def step_given_output_folders_are_empty(context, dataset):
    base_dir = f"runs/{dataset}"

    for dirpath, dirnames, _ in os.walk(base_dir):
        for name in dirnames:
            if context.pipeline_vars["run_identifier"] in name:
                rmtree(os.path.join(dirpath, name))
                print(f"Removed {os.path.join(dirpath, name)}")

@given("the pipeline variables are saved to config")
def step_then_pipeline_vars_saved_to_config(context):
    template_name = context.pipeline_vars.get("template_name")
    pipeline_name = context.pipeline_vars.get("pipeline_name")

    if not template_name:
        raise AssertionError("Missing required pipeline variable: template_name")
    if not pipeline_name:
        raise AssertionError("Missing required pipeline variable: pipeline_name")

    config_dir = Path("experiments") / "configs" / "templates" / template_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a copy for saving, excluding variables that are not part of the pipeline config
    # prev_pipeline is used to specify dependencies between pipelines but not part of the pipeline config
    ignore_param=['prev_pipeline',"run_identifier"]
    config_to_save = {k: v for k, v in context.pipeline_vars.items() if k not in ignore_param}

    config_path = config_dir / f"{pipeline_name}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_save, f, sort_keys=False)
    