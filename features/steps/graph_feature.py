from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then
import yaml


@given("a new graph pipeline with input from json file initialized with dataset test_data, file {file}, feature extractor {fe_name}, model {model}")
def step_given_new_graph_json_afterimage_model(context, file, fe_name, model):

    
    # get fe_attrs
    saved_file=f"test_data/{fe_name}/saved_components/pipeline/{file}/PacketReader->{fe_name}->JsonFeatureBuffer.yaml"
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
    fe_attrs=manifest["attrs"]["manifest"]["feature_extractor"]["attrs"]
    
    template=get_template("graph_feature_detection", dataset_name="test_data", file_name=file, model_name=model, fe_name=fe_name, fe_attrs=fe_attrs, graph_period=100)
    context.pipeline=Pipeline.load(template)
    context.config.userdata['benign_path']=context.pipeline.setup()
    
@given("a loaded graph pipeline with input from json file initialized with dataset test_data, file {file}, feature extractor {fe_name}, model {model}")
def step_given_loaded_graph_json_afterimage_model(context, file, fe_name, model):

    
    with open(context.config.userdata['benign_path']) as f:
        manifest = yaml.safe_load(f)
        
    manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
    
    context.pipeline=Pipeline.load(manifest)
    context.pipeline.on_load()