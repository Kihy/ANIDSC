from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then


import os
import yaml



@given("a new basic pipeline with input from csv file initialized with dataset test_data, file {file}, feature extractor {fe_name} and model {model}")
def step_given_new_csv_afterimage_model(context, file, fe_name, model):

    # get fe_attrs
    saved_file=f"test_data/{fe_name}/saved_components/pipeline/{file}/PacketReader->{fe_name}->TabularFeatureBuffer(256).yaml"
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
    fe_attrs=manifest["attrs"]["manifest"]["feature_extractor"]["attrs"]
    
    template=get_template("detection", dataset_name="test_data", file_name=file, model_name=model, fe_name=fe_name, fe_attrs=fe_attrs)
    
    context.pipeline=Pipeline.load(template)
    context.pipeline.setup()
    
    
@given("a loaded basic pipeline with input from csv file initialized with dataset test_data, file {file}, feature extractor {fe_name} and model {model}")
def step_given_loaded_csv_afterimage_model(context, file, fe_name, model):
    if model=="BoxPlot":
        saved_file=f"test_data/{fe_name}/saved_components/pipeline/benign_lenovo_bulb/CSVReader->LivePercentile->{model}->BaseEvaluator.yaml"
    
    else:
        saved_file=f"test_data/{fe_name}/saved_components/pipeline/benign_lenovo_bulb/CSVReader->LivePercentile->OnlineOD({model})->BaseEvaluator.yaml"
    
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
        
    manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
    
    context.pipeline=Pipeline.load(manifest)
    context.pipeline.on_load()
    



@then("the pipeline should not fail")
def step_then_data_processed_correctly(context):
    # the pipeline should run
    assert context.failed is False


@then("the components are saved")
def step_then_components_are_saved(context):
    manifest_path=context.pipeline.get_save_path()
    
    loaded_pipeline=Pipeline.load(manifest_path)

    loaded_pipeline.on_load() # load the components
    
    assert loaded_pipeline==context.pipeline
    
    

