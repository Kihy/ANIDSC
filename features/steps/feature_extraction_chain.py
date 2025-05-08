from behave import given, when, then
import yaml
from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from shutil import rmtree
import os


@given("a new {fe_class} feature extraction pipeline initialized with test_data dataset and file {file}")
def step_given_new_afterimage_and_file(context, fe_class, file):
    template=get_template("feature_extraction", dataset_name="test_data", file_name=file, fe_class=fe_class, save_buffer=True)
        
    context.pipeline=Pipeline.load(template)

@given("a loaded {fe_class} feature extraction pipeline initialized with test_data dataset and file {file}")
def step_given_loaded_afterimage_and_file(context, fe_class, file):
    saved_file=f"test_data/{fe_class}/saved_components/pipeline/benign_lenovo_bulb/PacketReader->{fe_class}->TabularFeatureBuffer(256).yaml"    
    
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
        
    manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
    
    
    context.pipeline=Pipeline.load(manifest)

@when("the pipeline starts")
def step_when_pipeline_starts(context):
    context.pipeline.start()
    
@given("{dataset} {fe_name} folder is empty")
def step_given_output_folder_is_empty(context, dataset, fe_name):
    
    file_dir = f"{dataset}/{fe_name}"
    if os.path.exists(file_dir):
        rmtree(file_dir)

