from datetime import datetime
import json
from ANIDSC.pipeline import Pipeline
from ANIDSC.utils.run_script import run_file, tune_hyperparameters
from behave import given, when, then
from shutil import rmtree
import fsspec
import pandas as pd
import os
from ANIDSC.utils.dataset_registry import dataset_registry

@given("Pipeline variable: {var_name} -> {var_value}")
def step_given_pipeline_var(context, var_name, var_value):
    context.pipeline_vars[var_name] = _parse_value(var_value)

def _parse_value(value: str):
    """Try to parse as JSON, fall back to raw string."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
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
    context.pipeline_vars["run_identifier"] = context.run_identifier

@then("the optuna database is created")
def step_then_optuna_db_created(context):
    dataset=context.file_iterator.location
    pipeline_name=context.pipeline_vars["pipeline_name"]
    run_id=context.pipeline_vars["run_identifier"]
    db_path = f"runs/{dataset}/{pipeline_name}/{run_id}/optuna.db"
    assert os.path.exists(db_path), f"Optuna database {db_path} does not exist"

@given("a {pipeline_name} pipeline")
def step_given_pipeline(context, pipeline_name):

    context.pipeline_vars["pipeline_name"] = pipeline_name


@when("the pipeline starts")
def step_when_pipeline_starts(context):
    context.pipelines, summary = run_file(
        context.file_iterator, context.pipeline_vars, return_pipeline=True
    )

@when("the pipeline starts with optuna tuning enabled")
def step_when_pipeline_starts_with_optuna(context):
    study = tune_hyperparameters(
        context.file_iterator, context.pipeline_vars, context.tuning_vars
    )



@given("folders in {dataset} with {run_identifier} are empty")
def step_given_output_folders_are_empty(context, dataset, run_identifier):
    base_dir = f"runs/{dataset}"

    for dirpath, dirnames, _ in os.walk(base_dir):
        for name in dirnames:
            if run_identifier in name:
                rmtree(os.path.join(dirpath, name))
                print(f"Removed {os.path.join(dirpath, name)}")
