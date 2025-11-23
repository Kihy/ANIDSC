from datetime import datetime
from ANIDSC.pipeline import Pipeline
from ANIDSC.utils.run_script import run_file
from behave import given, when, then
from shutil import rmtree
import fsspec
import pandas as pd
import os


   

@given("Model: {model_name}")
def step_given_feature_extractor(context, model_name):
    context.pipeline_vars["model_name"]=model_name 
    
@given("Graph Rep: {graph_rep}")
def step_given_feature_extractor(context, graph_rep):
    context.pipeline_vars["graph_rep"]=graph_rep 


@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the results are written")
def step_then_results_saved(context):
    for pipeline in context.pipelines:
        
        p_name=pipeline.name.split("->")
        for i in p_name:
            if i.endswith("ResultWriter"):
                results_path=str(pipeline.get_attr_by_name(i,"feature_path"))
                if results_path.endswith("csv.zst"):
                    with fsspec.open(results_path, "rt", compression="zstd") as f:
                        csv_df=pd.read_csv(f)
                        assert len(csv_df)>0, f"{results_path} is empty"
                elif results_path.endswith("ndjson.zst"):
                    assert os.stat(results_path).st_size > 0, f"{results_path} is empty"
                        
        
        


@then("the components are saved")
def step_then_components_are_saved(context):
    
    for pipeline in context.pipelines:
        manifest_path=str(pipeline.save_path)
        
        loaded_pipeline=Pipeline.load(manifest_path)
        
        # do not call setup because attributes in setup are assumed to be created dynamically
                
        assert loaded_pipeline==pipeline
    


@given("The test_data file iterator")
def step_given_dataset_and_file(context):

    def iterate_files():
        yield "new", "benign_lenovo_bulb", "datasets/test_data"

        # Attack files (subsequent runs)
        for attack in [
            "malicious_ACK_Flooding",
            "malicious_Port_Scanning",
            "malicious_Service_Detection"
        ]:

            yield "loaded", attack, "datasets/test_data"

    context.file_iterator = iterate_files
    context.pipeline_vars = {}
    context.pipeline_vars["run_identifier"] = context.run_identifier


@given("Meta Extractor: {meta_extractor}")
def step_given_feature_extractor(context, meta_extractor):
    context.pipeline_vars["meta_extractor"] = meta_extractor


@given("Data Source: {reader_type}")
def step_given_feature_extractor(context, reader_type):
    context.pipeline_vars["reader_type"] = reader_type
    
@given("FE Name: {fe_name}")
def step_given_feature_extractor(context, fe_name):
    context.pipeline_vars["fe_name"] = fe_name


@given("Feature Extractor: {feature_extractor}")
def step_given_feature_extractor(context, feature_extractor):
    context.pipeline_vars["feature_extractor"] = feature_extractor

@given("a {pipeline_name} pipeline")
def step_given_pipeline(context, pipeline_name):

    context.pipeline_name = pipeline_name


@when("the pipeline starts")
def step_when_pipeline_starts(context):
    
    context.pipelines=run_file(context.file_iterator, context.pipeline_name, context.pipeline_vars)


@given("{dataset} {fe_name} folder is empty")
def step_given_output_folder_is_empty(context, dataset, fe_name):

    file_dir = f"datasets/{dataset}/{fe_name}"
    if os.path.exists(file_dir):
        rmtree(file_dir)
        
@given("folders that starts with {dataset} {fe_name} are empty")
def step_given_output_folders_are_empty(context, dataset, fe_name):
    base_dir = f"datasets/{dataset}"

    if not os.path.exists(base_dir):
        return

    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)

        # remove only directories whose names start with fe_name
        if os.path.isdir(path) and name.startswith(fe_name):
            rmtree(path)
