from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then


import os
import yaml


@given("Feature Extracted by {meta_extractor} and {feature_extractor}")
def step_given_feature_extractor(context, meta_extractor, feature_extractor):
    context.custom_vars['feature_extractor']=feature_extractor
   

@given("Model: {model_name}")
def step_given_feature_extractor(context, model_name):
    context.custom_vars["model_name"]=model_name 


@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the components are saved")
def step_then_components_are_saved(context):
    manifest_path=str(context.pipeline.save_path)
    
    loaded_pipeline=Pipeline.load(manifest_path)

    loaded_pipeline.setup() # load the components
    
    assert loaded_pipeline==context.pipeline
    
    

