from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then


import os
import yaml


@given("a new graph pipeline with input from csv file initialized with dataset test_data, file {file}, feature extractor {fe_name}, model {model}, node encoder {node_encoder}")
def step_given_new_graph_csv_afterimage_model(context, file, fe_name, model, node_encoder):

    
    # get fe_attrs
    saved_file=f"test_data/{fe_name}/saved_components/pipeline/{file}/PacketReader->{fe_name}->TabularFeatureBuffer(256).yaml"
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
    fe_attrs=manifest["attrs"]["manifest"]["feature_extractor"]["attrs"]
    
    
    context.pipeline=Pipeline.load(get_template("graph_detection", dataset_name="test_data", file_name=file, model_name=model, fe_name=fe_name, fe_attrs=fe_attrs, node_encoder=node_encoder))
    context.pipeline.setup()
    
@given("a loaded graph pipeline with input from csv file initialized with dataset test_data, file {file}, feature extractor {fe_name}, model {model}, node encoder {node_encoder}")
def step_given_loaded_graph_csv_afterimage_model(context, file, fe_name, model, node_encoder):
    saved_file=f"test_data/{fe_name}/saved_components/pipeline/benign_lenovo_bulb/CSVReader->MultilayerSplitter({node_encoder}->{model}).yaml"
    
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
        
    manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
    
    context.pipeline=Pipeline.load(manifest)
    context.pipeline.on_load()
    