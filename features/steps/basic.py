from ANIDSC.pipeline import Pipeline
from ANIDSC.utils.run_script import run_file
from behave import given, when, then
from shutil import rmtree


import os
import yaml


@given("Feature Extracted by {meta_extractor} and {feature_extractor}")
def step_given_feature_extractor(context, meta_extractor, feature_extractor):
    context.pipeline_vars['feature_extractor']=feature_extractor
    context.pipeline_vars['meta_extractor']=meta_extractor
   
   

@given("Model: {model_name}")
def step_given_feature_extractor(context, model_name):
    context.pipeline_vars["model_name"]=model_name 


@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the components are saved")
def step_then_components_are_saved(context):
    
    for pipeline in context.pipelines:
        manifest_path=str(pipeline.save_path)
        
        loaded_pipeline=Pipeline.load(manifest_path)

        loaded_pipeline.setup() # load the components
        
        assert loaded_pipeline==pipeline
    


@given("The test_data file iterator")
def step_given_dataset_and_file(context):

    def iterate_files():
        yield "new", "benign_lenovo_bulb"

        # Attack files (subsequent runs)
        for attack in [
            "malicious_ACK_Flooding",
            "malicious_Port_Scanning",
            "malicious_Service_Detection",
        ]:

            yield "loaded", attack

    context.file_iterator = iterate_files
    context.pipeline_vars = {"dataset_name": "test_data"}


@given("Meta Extractor: {meta_extractor} and Feature Extractor: {feature_extractor}")
def step_given_feature_extractor(context, meta_extractor, feature_extractor):

    context.pipeline_vars["feature_extractor"] = feature_extractor
    context.pipeline_vars["meta_extractor"] = meta_extractor


@given("a {pipeline_name} pipeline")
def step_given_pipeline(context, pipeline_name):

    context.pipeline_name = pipeline_name


@when("the pipeline starts")
def step_when_pipeline_starts(context):
    context.pipelines=run_file(context.file_iterator, context.pipeline_name, context.pipeline_vars)


@given("{dataset} {fe_name} folder is empty")
def step_given_output_folder_is_empty(context, dataset, fe_name):

    file_dir = f"{dataset}/{fe_name}"
    if os.path.exists(file_dir):
        rmtree(file_dir)


