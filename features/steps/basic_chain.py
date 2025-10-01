from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then


import os
import yaml


@given("Feature Extracted by {meta_extractor} and {feature_extractor}")
def step_given_feature_extractor(context, meta_extractor, feature_extractor):
    context.custom_vars['feature_extractor']=feature_extractor
    # get fe_attrs
    saved_file=f"{context.custom_vars['dataset_name']}/{feature_extractor}/saved_components/pipeline/{context.custom_vars['file_name']}/PacketReader->{meta_extractor}->{feature_extractor}.yaml"
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
    fe_attrs=manifest["attrs"]["manifest"]["feature_extractor"]["attrs"]
    context.custom_vars['fe_attrs']=fe_attrs

@given("Model: {model_name}")
def step_given_feature_extractor(context, model_name):
    context.custom_vars["model_name"]=model_name 


@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the components are saved")
def step_then_components_are_saved(context):
    manifest_path=context.pipeline.get_save_path()
    
    loaded_pipeline=Pipeline.load(manifest_path)

    loaded_pipeline.setup() # load the components
    
    assert loaded_pipeline==context.pipeline
    
    

