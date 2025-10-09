from behave import given, when, then
import yaml
from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from shutil import rmtree
import os

@given("Dataset: {dataset_name} and File: {file_name}")
def step_given_dataset_and_file(context, dataset_name, file_name):
    context.custom_vars={}
    context.custom_vars["dataset_name"]=dataset_name 
    context.custom_vars["file_name"]=file_name

@given("Meta Extractor: {meta_extractor} and Feature Extractor: {feature_extractor}")
def step_given_feature_extractor(context, meta_extractor, feature_extractor):
    
    context.custom_vars["feature_extractor"]=feature_extractor 
    context.custom_vars["meta_extractor"]=meta_extractor
    


@given("a {state} {pipeline_name} pipeline")
def step_given_pipeline(context, state, pipeline_name):
    
    
    if state=="new":
        
        context.pipeline=Pipeline.load(get_template(pipeline_name, **context.custom_vars))
        context.pipeline.setup()
        
        
        context.config.userdata['benign_path']=context.pipeline.save_path

    elif state=="loaded":
        
        with open(context.config.userdata['benign_path']) as f:
            manifest = yaml.safe_load(f)
        
        # datasource is always 0
        manifest["attrs"]["manifest"][0]["attrs"]["file_name"]=context.custom_vars["file_name"]
        manifest["attrs"]["manifest"][0]["file"]=None
        
        context.pipeline=Pipeline.load(manifest)
        context.pipeline.setup()



@when("the pipeline starts")
def step_when_pipeline_starts(context):
    context.pipeline.start()
    
@given("{dataset} {fe_name} folder is empty")
def step_given_output_folder_is_empty(context, dataset, fe_name):
    
    file_dir = f"{dataset}/{fe_name}"
    if os.path.exists(file_dir):
        rmtree(file_dir)

