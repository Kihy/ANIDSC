from behave import given, when, then
import yaml
from ANIDSC.component.pipeline_component import Pipeline
from ANIDSC.templates import get_template

@given("a new afterimage feature extraction pipeline initialized with test_data dataset and file {file}")
def step_given_afterimage_and_file(context, file):
    template=get_template("feature_extraction")
    context.pipeline=Pipeline.load(template.format("test_data",file))
    
@given("a loaded afterimage feature extraction pipeline initialized with test_data dataset and file {file}")
def step_given_afterimage_and_file(context, file):
    saved_file=f"test_data/AfterImage/manifest/benign_lenovo_bulb/PacketReader->AfterImage->TabularFeatureBuffer(256).yaml"
    
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
        
    manifest["components"]["data_source"]["file_name"]=file
    
    
    context.pipeline=Pipeline.load(manifest)

@when("the pipeline starts")
def step_when_pipeline_starts(context):
    context.pipeline.process()